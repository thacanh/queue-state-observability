"""Run checked, resumable fixed-penalty controls on an authorized CUDA runtime."""

from argparse import ArgumentParser
from functools import partial
from hashlib import sha256
from itertools import product
from pathlib import Path
import json
import os
import signal
import subprocess
import threading
import time

import numba
import numpy as np
import pandas as pd
import psutil
import torch

from deadline_guard import _guard_service_bank
from estimated_ddqn import sha, source_inputs, save_metadata
from estimated_queue import EstimatedQueue
from rl_core_execution import BatchQueue, atomic_torch_save, train_dynamic
from rl_pilot import prepare_stage2_config


class SharedCalendar(EstimatedQueue):
    def __init__(self, *args, observation, **kwargs):
        self.observation = observation
        super().__init__(*args, **kwargs)

    def residual(self, arrivals):
        predicted = super().residual(arrivals)
        if self.observation == "estimated":
            return predicted
        # The same Numba dispatch state is maintained for both observers.
        truth = torch.as_tensor(self.truth, dtype=torch.float32, device=self.device)
        return (truth - arrivals[:, None, None]).clamp_min(0)


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def runtime():
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU. Select an available free GPU runtime before training.")
    card = torch.cuda.get_device_properties(0)
    if not ("T4" in card.name or card.total_memory >= 14*1024**3):
        raise RuntimeError(f"Unvalidated GPU: {card.name}")
    try:
        driver = subprocess.check_output(["nvidia-smi","--query-gpu=driver_version",
                                          "--format=csv,noheader"],text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        driver = None
    return {"gpu":card.name,"total_vram_bytes":card.total_memory,"driver":driver,
            "torch":str(torch.__version__),"cuda":str(torch.version.cuda),"numpy":np.__version__,
            "pandas":pd.__version__,"numba":numba.__version__}


def monitor(stop, samples):
    process = psutil.Process()
    while not stop.is_set():
        sample = {"monotonic_s":time.monotonic(),"rss_bytes":process.memory_info().rss}
        try:
            line = subprocess.check_output(["nvidia-smi","--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits"],text=True, timeout=2).splitlines()[0]
            utilization, memory = [float(v.strip()) for v in line.split(",")]
            sample.update(gpu_util_pct=utilization,gpu_memory_mib=memory)
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            pass
        samples.append(sample)
        stop.wait(1.)


def schedule_budget(phase):
    return {"T2":2400,"T4":1800,"T8":900}[phase]


def verify_pilot(record, historical, seed, lam):
    required = {"method":"fixed","seed":seed,"fixed_lambda":lam,"stage2_transitions":2_000_000,
                "realized_transitions":2_000_128,"actual_optimizer_updates":7695,
                "holdout_used":False,"train_days":"d01-d07"}
    if any(record.get(k)!=v for k,v in required.items()):
        raise ValueError(f"Historical fixed-family config differs for seed {seed}, lambda {lam}")
    if any(record[k]!=historical[k] for k in ("source_sha256","input_sha256")):
        raise ValueError("Historical training source or starting inputs differ")


def publish_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    save_metadata(path,data)


def run_job(folder, spec, inputs, factory, remaining):
    marker=folder/"status.json"
    final=folder/"final.pt"
    identity=fingerprint(spec)
    previous=json.loads(marker.read_text()) if marker.is_file() else {}
    if previous and previous["identity"]!=identity:
        raise ValueError(f"Unsafe resume: input/config/software changed for {folder.name}")
    if previous.get("status")=="COMPLETE":
        assert sha(final)==previous["model_sha256"]
        return 0.
    config=spec["config"]|{"identity":identity}
    trace, validation, scales, initial=inputs
    clock=time.monotonic()
    cpu=time.process_time()
    stop=threading.Event()
    samples=[]
    poller=threading.Thread(target=monitor,args=(stop,samples),daemon=True)
    torch.cuda.reset_peak_memory_stats()

    def budget_hit(*_):
        raise TimeoutError("Active GPU training budget expired; retain last atomic checkpoint")

    signal.signal(signal.SIGALRM,budget_hit)
    signal.alarm(max(1,int(remaining)))
    publish_json(marker,{"status":"RUNNING","identity":identity,**spec,
                         "total_active_seconds":previous.get("total_active_seconds",0)})
    poller.start()
    status="FAILED"
    metrics={}
    try:
        result, meta=train_dynamic(trace,validation,scales,config,folder/"latest.pt",
                                  initial=initial,resume=True,queue_factory=factory)
        realized=meta["steps"]*config["nenv"]
        assert (realized,meta["updates"])==(2_000_128,7695)
        atomic_torch_save({"model":result,"scales":scales,"identity":identity,"config":config,"meta":meta},final)
        metrics.update(model_sha256=sha(final),actual_transitions=realized,optimizer_updates=meta["updates"])
        status="COMPLETE"
    except BaseException as error:
        metrics["error"]=f"{type(error).__name__}: {error}"
        raise
    finally:
        signal.alarm(0)
        stop.set()
        poller.join(timeout=2.)
        elapsed=time.monotonic()-clock
        publish_json(marker,{"status":status,"identity":identity,**spec,**metrics,
                    "active_s":elapsed,"total_active_seconds":elapsed+previous.get("total_active_seconds",0),
                    "cpu_seconds":time.process_time()-cpu,"peak_allocated_cuda_bytes":torch.cuda.max_memory_allocated(),
                    "peak_reserved_cuda_bytes":torch.cuda.max_memory_reserved(),"resource_samples":samples})
    return elapsed


def execute(phase, workspace, project, output):
    registry_path=workspace/"registry.json"
    registry=json.loads(registry_path.read_text())["jobs"]
    train_path,val_path, historical=source_inputs(project,registry_path,(42,43))
    out=output/phase
    out.mkdir(parents=True,exist_ok=True)
    env=runtime()
    print(json.dumps(env),flush=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    train,val=pd.read_csv(train_path),pd.read_csv(val_path)
    bank_names,bank,offsets=_guard_service_bank(train_path)
    t8=None
    if phase=="T8":
        prefix=workspace/"prefix_inputs"
        manifest=json.loads((prefix/"manifest.json").read_text())
        trace=prefix/"prefix_train.csv"
        archive=prefix/"prefix_bank.npz"
        calibration_path=prefix/"calibration.json"
        assert sha(trace)==manifest["training_csv_sha256"] and sha(archive)==manifest["training_bank_sha256"]
        t8=json.loads(calibration_path.read_text())
        assert t8["fit_only_elapsed_0_to_6"] is True
        with np.load(archive,allow_pickle=False) as data:
            bank_names={key:j for j,key in enumerate(data["functions"].tolist())}
            bank,offsets=data["bank"],data["offsets"]
        train=pd.read_csv(trace)
        val=train.iloc[:min(10000,len(train))]
        prefix_identity={"train":sha(trace),"bank":sha(archive),"normalization":sha(calibration_path)}
    source={p.name:sha(p) for p in Path(__file__).parent.glob("*.py")}
    jobs=list(product((42,43),("oracle","estimated"),(2,))) if phase=="T4" else (
        list(product((42,43),("oracle",),(0,5,20,80))) if phase=="T2" else
        list(product((42,43),("estimated",),(2,))))
    protocol={"phase":phase,"jobs":jobs,"source":source,"registry_sha256":sha(registry_path),
              "runtime":env,"training_budget":dict(transitions=2_000_128,updates=7695,
              active_session_cap_s=schedule_budget(phase)),
              "evaluation":"Use the predeclared workloads. No evaluation outcomes enter training."}
    json_path=out/"training_protocol.json"
    if json_path.exists():
        assert fingerprint(json.loads(json_path.read_text()))==fingerprint(protocol)
    else:
        publish_json(json_path,protocol)
    consumed=0.
    for marker in out.glob("*/status.json"):
        consumed+=json.loads(marker.read_text()).get("total_active_seconds",0)
    for seed,observation,lam in jobs:
        primary, reference=historical[seed]
        init_path=primary
        if phase=="T2":
            record=registry[f"fixed_seed{seed}_lambda{lam}"]["record"]
            verify_pilot(record,reference,seed,lam)
            recovered=project/record["model_path"]
            if recovered.is_file() and sha(recovered)==record["model_sha256"]:
                folder=out/f"oracle_{seed}_lambda{lam}"
                folder.mkdir(exist_ok=True)
                final=folder/"final.pt"
                if not final.exists():
                    temp=folder/"recovered.tmp"
                    temp.write_bytes(recovered.read_bytes())
                    os.replace(temp,final)
                assert sha(final)==record["model_sha256"]
                publish_json(folder/"status.json",{"status":"COMPLETE","origin":"recovered historical checkpoint",
                    "model_sha256":sha(final),"seed":seed,"lambda":lam,"history":record,"identity":record["identity"]})
                continue
        if phase=="T8":
            init_path=workspace/"estimated"/f"seed{seed}.pt"
            expected=json.loads((workspace/"temporal_protocol.json").read_text())["models"][f"estimated/{seed}"]["sha256"]
            assert sha(init_path)==expected
        stage=torch.load(init_path,map_location="cpu",weights_only=True)
        scales=stage["scales"].copy()
        if phase=="T8":
            scales.update(t8["scales"])
        config=prepare_stage2_config(seed,"fixed",lam,"cuda","pending")
        config.update(checkpoint_every=500 if phase=="T2" else 1000,
                      amp_init_scale=1024.,amp_growth_interval=1_000_000_000)
        factory=(BatchQueue if phase=="T2" else partial(SharedCalendar,observation=observation,
                    names=bank_names,bank=bank,offsets=offsets))
        spec={"phase":phase,"origin":"new training","seed":seed,"lambda":lam,"observation":observation,
              "calendar":"Torch CUDA" if phase=="T2" else "shared Numba float32","runtime":env,
              "protocol":fingerprint(protocol),"initial_checkpoint_sha256":sha(init_path),
              "input_identity":reference["input_sha256"] if phase!="T8" else prefix_identity,
              "config":config,"guard_in_training":False}
        remaining=schedule_budget(phase)-consumed
        if remaining<5:
            raise TimeoutError(f"GPU budget exhausted for {phase}")
        folder=out/(f"{observation}_{seed}_lambda{lam}")
        print(f"{phase}: {folder.name}, {remaining:.1f}s active budget",flush=True)
        consumed+=run_job(folder,spec,(train,val,scales,stage["model"]),factory,remaining)


if __name__=="__main__":
    cli=ArgumentParser(description=__doc__)
    cli.add_argument("phase",choices=["T2","T4","T8"])
    cli.add_argument("--workspace",type=Path,required=True)
    cli.add_argument("--project",type=Path,required=True)
    cli.add_argument("--output",type=Path,required=True)
    params=cli.parse_args()
    execute(params.phase,params.workspace,params.project,params.output)
