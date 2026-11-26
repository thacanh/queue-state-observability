import tempfile
import unittest
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from estimated_queue import EstimatedQueue
from rl_core_execution import BatchQueue, train_dynamic
from rl_pilot import prepare_stage2_config
from routing_simulator import replay

SCALES = {"p50": 1., "p99": 1., "unc": 10., "ia": 1., "res": 1., "lat_ref": .25}
SPEEDS, WORKERS = [1.5, 1., .7], [4, 3, 2]
NAMES, BANK, OFFSETS = {"a": 0, "b": 1}, np.array([.25, 1., 4., .5, 1., 2.]), np.array([0, 3, 6])
FACTORY = partial(EstimatedQueue, names=NAMES, bank=BANK, offsets=OFFSETS)


def trace():
    random = np.random.default_rng(42)
    size = 320
    return pd.DataFrame({
        "arrival": np.cumsum(random.integers(0, 5, size)) / 16,
        "est_p50": np.ones(size),
        "est_p99": np.full(size, 4.),
        "service_demand": random.integers(1, 30, size) / 16,
        "slo": np.ones(size),
        "func_id": np.where(np.arange(size) % 2, "a", "b"),
    })


def queue(factory, source, count=4, window=41):
    return factory(source, SPEEDS, WORKERS, SCALES, count, window, torch.device("cpu"),
                   torch.Generator().manual_seed(33), "full")


class EstimatedQueueTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_dispatch_and_reward_match_with_identical_actions(self):
        frame = trace()
        oracle, estimate = (queue(kind, frame) for kind in (BatchQueue, FACTORY))
        random = torch.Generator().manual_seed(99)
        for _ in range(350):
            actions = torch.randint(3, (4,), generator=random)
            a, b = oracle.step(actions), estimate.step(actions)
            self.assertTrue(all(torch.equal(a[i], b[i]) for i in (1, 2, 3)))
            self.assertTrue(torch.equal(oracle.idx, estimate.idx))

    def test_hidden_duration_cannot_change_observation_before_completion(self):
        base = trace()
        base.loc[:, "service_demand"] = 3.
        longer = base.assign(service_demand=5.)
        a, b = queue(FACTORY, base, 1), queue(FACTORY, longer, 1)
        self.assertTrue(torch.equal(a.observe(), b.observe()))
        for _ in range(8):
            oa, la, *_ = a.step(torch.zeros(1, dtype=torch.long))
            ob, lb, *_ = b.step(torch.zeros(1, dtype=torch.long))
            self.assertNotEqual(float(la), float(lb))
            self.assertTrue(torch.equal(oa, ob))

    def test_estimated_wait_matches_frozen_replay(self):
        frame = trace()
        env = queue(FACTORY, frame, 1, 80)
        ix = int(env.idx[0])
        frame = frame.iloc[ix:ix+80]
        exact = frame[["arrival", "est_p50", "est_p99", "service_demand", "slo"]].to_numpy()
        groups = frame["func_id"].map(NAMES).to_numpy(np.int64)
        weights = tuple(np.zeros(shape, np.float32) for shape in
                        ((256, 22), (256,), (256, 256), (256,), (3, 256), (3,)))
        weights[-1][1] = 1.
        prefix = np.r_[0., np.cumsum(BANK)]
        means = np.diff(prefix[OFFSETS]) / np.diff(OFFSETS)
        scales = np.array([SCALES[k] for k in ("p50", "p99", "unc", "ia", "res")])
        result = replay(exact, groups, exact[:, 1].copy(), np.array(SPEEDS),
                        np.array(WORKERS), scales, weights, 0, False, True, 1.,
                        BANK, OFFSETS, prefix, means)
        for i in range(80):
            wait = float(env.residual(None)[0, 1].min())
            self.assertAlmostEqual(wait, result[i, 5] + result[i, 7], places=5)
            _, latency, *_ = env.step(torch.ones(1, dtype=torch.long))
            self.assertAlmostEqual(float(latency), result[i, 0], places=5)

    def test_resume_includes_event_state_and_optimizer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = trace()
            conf = prepare_stage2_config(42, "fixed", 2., "cpu", "resume-test", 960)
            conf.update(nenv=4, buffer=200, batch=32, episode_len=41,
                        dual_envs=2, learning_starts=32, checkpoint_every=0)
            uninterrupted, meta = train_dynamic(frame, frame, SCALES, conf, root/"complete.pt",
                                                queue_factory=FACTORY)
            _, stopped = train_dynamic(frame, frame, SCALES, conf, root/"resume.pt",
                                       queue_factory=FACTORY, stop_after_batches=103)
            resumed, new_meta = train_dynamic(frame, frame, SCALES, conf, root/"resume.pt",
                                             queue_factory=FACTORY, resume=True)
            self.assertTrue(stopped["interrupted"])
            self.assertEqual(meta, new_meta)
            self.assertTrue(all(torch.equal(resumed[k], v) for k, v in uninterrupted.items()))

    def test_fixed_training_updates_do_not_depend_on_validation(self):
        train = trace()
        shifted_validation = train.assign(slo=np.full(len(train), .001),
                                          service_demand=np.full(len(train), 120.))
        config = prepare_stage2_config(42, "fixed", 2., "cpu", "validation-isolation", 192)
        config.update(nenv=4, buffer=128, batch=16, episode_len=19,
                      dual_envs=2, learning_starts=16, checkpoint_every=0)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, original_meta = train_dynamic(train, train, SCALES, config, root/"original.pt",
                                                     queue_factory=FACTORY)
            shifted, shifted_meta = train_dynamic(train, shifted_validation, SCALES, config, root/"shifted.pt",
                                                   queue_factory=FACTORY)
        self.assertEqual(original_meta, shifted_meta)
        self.assertEqual(original_meta["history"], [])
        for name, value in original.items():
            self.assertTrue(torch.equal(value, shifted[name]))


if __name__ == "__main__":
    unittest.main()
