"""Exercise the shared calendar before GPU training."""

import tempfile
import unittest
from functools import partial
from pathlib import Path

import numpy as np
import torch

from gpu_training import SharedCalendar
from rl_core_execution import BatchQueue, train_dynamic
from rl_pilot import prepare_stage2_config
from test_estimated_queue import BANK, NAMES, OFFSETS, SCALES, EstimatedQueueTest, queue, trace


def factory(observation):
    return partial(SharedCalendar, observation=observation, names=NAMES, bank=BANK, offsets=OFFSETS)


class SharedCalendarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fixed_actions_preserve_events_and_match_legacy_reward(self):
        frame=trace()
        ref=queue(BatchQueue,frame)
        oracle,estimated=[queue(factory(view),frame) for view in ("oracle","estimated")]
        generator=torch.Generator().manual_seed(761)
        for k in range(88):
            # A constant server visits the deterministic worker tie-break path.
            actions=torch.zeros(4,dtype=torch.long) if k<42 else torch.randint(3,(4,),generator=generator)
            a=ref.step(actions)
            b=oracle.step(actions)
            c=estimated.step(actions)
            for i in (1,2,3):
                self.assertTrue(torch.equal(a[i],b[i]) and torch.equal(b[i],c[i]))
            for name in ("truth","heads","tails","pending","links","begins","ends"):
                np.testing.assert_array_equal(getattr(oracle,name),getattr(estimated,name))
            np.testing.assert_array_equal(oracle.truth,ref.avail.numpy())
            self.assertTrue(torch.equal(oracle.idx,estimated.idx))
            self.assertTrue(torch.equal(oracle.steps,estimated.steps))
            self.assertTrue(torch.equal(a[0],b[0]))
            np.testing.assert_array_equal(oracle.residual(oracle.d["a"][oracle.idx]),
                                          ref.residual(ref.d["a"][ref.idx]))

    def test_hidden_duration_visibility(self):
        frame=trace().assign(service_demand=3.)
        longer=frame.assign(service_demand=5.)
        actions=torch.zeros(1,dtype=torch.long)
        for mode in ("oracle","estimated"):
            a,b=queue(factory(mode),frame,1),queue(factory(mode),longer,1)
            # Preserve differences in the observed residual instead of clipping both at one.
            a.sc=b.sc=SCALES|{"res":100.}
            self.assertTrue(torch.equal(a.observe(),b.observe()))
            equal=[]
            for _ in range(8):
                obs_a,la,*_=a.step(actions)
                obs_b,lb,*_=b.step(actions)
                self.assertNotEqual(float(la),float(lb))
                equal.append(torch.equal(obs_a,obs_b))
            self.assertEqual(all(equal),mode=="estimated")

    def test_resume_same_state_and_applied_updates(self):
        frame=trace()
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)
            for mode in ("oracle","estimated"):
                config=prepare_stage2_config(42,"fixed",2.,"cpu",mode,240)
                config.update(nenv=4,buffer=120,batch=16,episode_len=41,
                              dual_envs=2,learning_starts=32,checkpoint_every=15)
                full,m1=train_dynamic(frame,frame,SCALES,config,path/f"{mode}_full.pt",
                                      queue_factory=factory(mode))
                _,interrupted=train_dynamic(frame,frame,SCALES,config,path/f"{mode}_resume.pt",
                                  stop_after_batches=29,queue_factory=factory(mode))
                resumed,m2=train_dynamic(frame,frame,SCALES,config,path/f"{mode}_resume.pt",resume=True,
                                         queue_factory=factory(mode))
                self.assertTrue(interrupted["interrupted"])
                self.assertEqual(m1,m2)
                self.assertTrue(all(torch.equal(v,resumed[k]) for k,v in full.items()))


if __name__=="__main__":
    unittest.main()
