def make_stage2_plan(seeds=(42,43), fixed_grid=(0,2,5,10,15,20,30,40,80)):
    """Predetermined pilot grid. Every job starts from the seed's same legacy stage-1 weights."""
    jobs=[]
    for seed in seeds:
        for method,lam in [('latency',0.),('adaptive',10.)]+[('fixed',float(x)) for x in fixed_grid]:
            jobs.append({'seed':int(seed),'method':method,'fixed_lambda':lam,'stage2_transitions':2_000_000,
                         'train_days':'d01-d07','validation_days':'d08-d09','holdout_used':False})
    return jobs

def hash_path(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def canonical_source_hash(notebook_path):
    """Hash the AST of E2's transitive code, excluding output, metadata and comments.

    Keep the explicit dependency list in sync when an E2 function starts using a
    new helper or global. Missing/duplicate definitions fail closed.
    """
    import ast
    names={'DynamicQNet','BatchQueue','atomic_torch_save','capture_rng','restore_rng',
           'replay_init','train_dynamic','evaluate_network','prepare_stage2_config',
           'hash_path','canonical_source_hash','stage2_job'}
    nb=json.loads(Path(notebook_path).read_text(encoding='utf8'))
    nodes={}
    for cell in nb.get('cells',[]):
        if cell.get('cell_type')!='code': continue
        tree=ast.parse(''.join(cell.get('source',[])))
        for node in tree.body:
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and node.name in names:
                if node.name in nodes: raise ValueError('duplicate E2 dependency: '+node.name)
                nodes[node.name]=ast.dump(node,annotate_fields=True,include_attributes=False)
    if set(nodes)!=names: raise ValueError('missing E2 dependencies: '+str(sorted(names-set(nodes))))
    data=json.dumps({'schema':'e2-source-ast-v1','nodes':nodes},sort_keys=True,separators=(',',':'))
    return hashlib.sha256(data.encode('utf8')).hexdigest()

def prepare_stage2_config(seed,method,fixed_lambda,device,identity,transitions=2_000_000):
    if method not in ('latency','fixed','adaptive'): raise ValueError('invalid method')
    return {'device':str(device),'speeds':[1.5,1.,.7],'workers':[4,3,2],'nenv':256,'episode_len':3000,
      'mask':'full','dual_envs':128,'dual_warmup':512,'dual_measure':512,'seed':int(seed),
      'buffer':600000,'lr':1e-4,'lambda_init':10.,'identity':identity,'transitions':int(transitions),
      'learning_starts':30000,'eps_start':.2,'eps_fraction':.25,'method':method,'fixed_lambda':float(fixed_lambda),
      'clip_reward':True,'gamma':.99,'batch':4096,'target_update':500,'dual_update':200,'ema':.5,
      'dual_lr':100.,'target':.005,'lambda_max':80.,'checkpoint_every':500}

def stage2_job(root,run_dir,job,device='cuda',resume=True):
    """Execute one pilot job on legacy training and validation traces, never holdout.

    Stage-2 methods share initial weights, optimizer settings and transition counts;
    legacy stage-1 optimizer/replay were not saved, so all stage-2 optimizers reset equally.
    This primitive is not the complete master scheduler or final confirmation study.
    """
    import time
    root=Path(root); run_dir=Path(run_dir); seed=int(job['seed']); method=job['method']; lam=float(job['fixed_lambda'])
    tr=root/'multiday_final_5seeds'/'traces'/'train_d01_d07.csv'
    va=root/'multiday_final_5seeds'/'traces'/'val_d08_d09.csv'
    ck=root/'multiday_final_5seeds'/'models'/f'latency_ddqn_seed{seed}.pt'
    for p in (tr,va,ck):
        if not p.is_file(): raise FileNotFoundError(p)
    source=torch.load(ck,map_location='cpu',weights_only=False)
    if source.get('seed')!=seed or 'model' not in source or 'scales' not in source: raise ValueError('incompatible stage-1 checkpoint')
    paths={'train':hash_path(tr),'validation':hash_path(va),'stage1':hash_path(ck)}
    template=prepare_stage2_config(seed,method,lam,device,identity='pending',transitions=job['stage2_transitions'])
    notebook=root/'LOAD_BALANCING_MASTER.ipynb'
    if not notebook.is_file(): raise FileNotFoundError(notebook)
    code_hash=canonical_source_hash(notebook)
    runtime={'torch':str(torch.__version__),'numpy':str(np.__version__),'cuda':str(torch.version.cuda)}
    identity=hashlib.sha256(json.dumps({'inputs':paths,'config':template,'code':code_hash,'runtime':runtime},sort_keys=True).encode()).hexdigest()
    config=prepare_stage2_config(seed,method,lam,device,identity,job['stage2_transitions'])
    label=f"{method}_seed{seed}_lambda{lam:g}_{identity[:12]}"; out=run_dir/label; out.mkdir(parents=True,exist_ok=True)
    result_path=out/'validation.json'; final_path=out/'final.pt'; latest=out/'latest.pt'
    if result_path.exists() and final_path.exists():
        result=json.loads(result_path.read_text())
        if result.get('identity')==identity and result.get('model_sha256')==hash_path(final_path): return result
    train=pd.read_csv(tr); val=pd.read_csv(va)
    if len(train)!=868593 or len(val)!=242585: raise ValueError('legacy trace counts differ')
    start=time.monotonic(); model,meta=train_dynamic(train,val,source['scales'],config,latest,initial=source['model'],resume=resume)
    if model is None: raise RuntimeError('training interrupted before completion')
    train_seconds=time.monotonic()-start
    atomic_torch_save({'model':model,'scales':source['scales'],'identity':identity,'config':config,'meta':meta},final_path)
    eval_start=time.monotonic(); result,_=evaluate_network(val,model,source['scales'],config['speeds'],config['workers'],device=device)
    result.update({'identity':identity,'model_sha256':hash_path(final_path),'input_sha256':paths,'source_sha256':code_hash,'runtime':runtime,'method':method,
                   'seed':seed,'fixed_lambda':lam,'session_train_seconds':train_seconds,'eval_seconds':time.monotonic()-eval_start,
                   'optimization_updates':meta['updates'],'lambda_final':meta['lambda'],'scope':'validation only',
                   'timing_note':'session_train_seconds excludes time in earlier interrupted sessions'})
    temp=result_path.with_suffix('.tmp'); temp.write_text(json.dumps(result,indent=2)); os.replace(temp,result_path)
    return result

def run_e2_pilot(root, run_dir, device='cuda', max_retries=1):
    """Run the fixed 22-job pilot; record failures and resume verified jobs.

    This only evaluates the historical validation trace. It is not the requested
    five-seed, multiple-reconstruction confirmation and does not access holdout.
    """
    import time, traceback
    root=Path(root); run_dir=Path(run_dir); run_dir.mkdir(parents=True,exist_ok=True)
    if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('E2 pilot requires an available CUDA runtime')
    plan=make_stage2_plan(); assert len(plan)==22
    registry=run_dir/'pilot_registry.json'
    if registry.exists():
        state=json.loads(registry.read_text(encoding='utf8'))
        if state.get('plan')!=plan: raise ValueError('existing pilot plan differs')
    else: state={'plan':plan,'jobs':{},'selection':'not_performed','scope':'pilot; validation only'}
    def save():
        tmp=registry.with_suffix('.tmp'); tmp.write_text(json.dumps(state,indent=2),encoding='utf8'); os.replace(tmp,registry)
    for job in plan:
        key=f"{job['method']}_seed{job['seed']}_lambda{job['fixed_lambda']:g}"
        record=state['jobs'].setdefault(key,{'attempts':0})
        for attempt in range(max_retries+1):
            record.update(status='running',started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
            record['attempts']+=1; save()
            try:
                # stage2_job verifies identity and final checkpoint checksum even for completed jobs.
                result=stage2_job(root,run_dir,job,device=device,resume=True)
                expected=((job['stage2_transitions']+255)//256)-((30000+255)//256)
                if result['optimization_updates']!=expected: raise RuntimeError('unexpected optimizer update count')
                record.update(status='complete',identity=result['identity'],result=result,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
                save(); print('E2 pilot complete',key,flush=True); break
            except Exception as exc:
                record.update(status='failed',error=str(exc),traceback=traceback.format_exc());save()
                if attempt==max_retries: raise RuntimeError('E2 pilot failed: '+key) from exc
    state['status']='pilot_complete'; save()
    return state
