"""Completion-event queue observations for matched DDQN fine-tuning."""

import numpy as np
import torch
from numba import njit

from rl_core_execution import BatchQueue
from routing_simulator import estimated_calendar


@njit(cache=True)
def assign(actions, indices, steps, arrivals, demands, speeds, workers, groups,
           means, truth, heads, tails, links, begins, ends, event_groups,
           waiting, pending):
    latency = np.empty(len(actions), np.float32)
    for env, action in enumerate(actions):
        ix, event = indices[env], steps[env]
        now = arrivals[ix]
        worker = np.argmin(truth[env, action, :workers[action]])
        start = max(now, truth[env, action, worker])
        end = start + demands[ix] / speeds[action]
        truth[env, action, worker] = end
        latency[env] = end - now
        begins[env, event] = start
        ends[env, event] = end
        group = groups[ix]
        event_groups[env, event] = group
        links[env, event] = -1
        if heads[env, action, worker] < 0:
            heads[env, action, worker] = event
        else:
            links[env, tails[env, action, worker]] = event
        tails[env, action, worker] = event
        waiting[env, event] = start > now
        if waiting[env, event]:
            pending[env, action] += means[group]
    return latency


@njit(cache=True)
def observe_calendar(indices, arrivals, speeds, workers, heads, links, begins, ends,
                     groups, waiting, pending, bank, offsets, prefix, means):
    calendar = np.full(heads.shape, np.inf)
    for env, ix in enumerate(indices):
        now = np.float64(arrivals[ix])
        for server, count in enumerate(workers):
            for worker in range(count):
                event = heads[env, server, worker]
                while event >= 0 and begins[env, event] <= now:
                    if waiting[env, event]:
                        pending[env, server] -= means[groups[env, event]]
                        waiting[env, event] = False
                    if ends[env, event] > now:
                        break
                    event = links[env, event]
                heads[env, server, worker] = event
        estimated_calendar(now, speeds, workers, heads[env], begins[env], groups[env],
                           pending[env], bank, offsets, prefix, means, 1., calendar[env])
        calendar[env] -= now
    return calendar.astype(np.float32)


class EstimatedQueue(BatchQueue):
    """Derive the observation from elapsed service and known queue membership."""

    event_state = ("truth", "heads", "tails", "links", "begins", "ends",
                   "event_groups", "waiting", "pending")

    def __init__(self, trace, speeds, workers, scales, nenv, episode_len, device,
                 generator, mask, *, names, bank, offsets):
        self.groups = trace["func_id"].astype(str).map(names).to_numpy(np.int64)
        self.bank, self.offsets = bank, offsets
        self.prefix = np.r_[0., np.cumsum(bank)]
        self.means = np.diff(self.prefix[offsets]) / np.diff(offsets)
        self.speeds32 = np.asarray(speeds, np.float32)
        self.speeds64 = np.asarray(speeds, np.float64)
        self.workers_np = np.asarray(workers, np.int64)
        self.arrivals = trace["arrival"].to_numpy(np.float32)
        self.demands = trace["service_demand"].to_numpy(np.float32)
        super().__init__(trace, speeds, workers, scales, nenv, episode_len, device,
                         generator, mask)
        dims, events = (nenv, self.m, self.maxw), (nenv, self.L)
        self.truth = np.empty(dims, np.float32)
        self.heads = np.empty(dims, np.int32)
        self.tails = np.empty(dims, np.int32)
        self.links = np.zeros(events, np.int32)
        self.begins = np.zeros(events, np.float32)
        self.ends = np.zeros(events, np.float32)
        self.event_groups = np.zeros(events, np.int32)
        self.waiting = np.zeros(events, np.bool_)
        self.pending = np.empty((nenv, self.m), np.float64)
        self._reset_events(np.arange(nenv))

    def _reset_events(self, selected):
        self.truth[selected] = self.avail[selected].cpu().numpy()
        self.heads[selected] = -1
        self.tails[selected] = -1
        self.pending[selected] = 0.

    def residual(self, arrivals):
        residual = observe_calendar(
            self.idx.cpu().numpy(), self.arrivals, self.speeds64, self.workers_np,
            self.heads, self.links, self.begins, self.ends, self.event_groups, self.waiting,
            self.pending, self.bank, self.offsets, self.prefix, self.means)
        return torch.as_tensor(residual, device=self.device)

    def step(self, action):
        command = torch.stack((action, self.idx, self.steps)).cpu().numpy()
        latency = assign(
            *command, self.arrivals, self.demands, self.speeds32, self.workers_np,
            self.groups, self.means, self.truth, self.heads, self.tails, self.links,
            self.begins, self.ends, self.event_groups, self.waiting, self.pending)
        latency = torch.as_tensor(latency, device=self.device)
        violation = (latency > self.d["slo"][self.idx]).float()
        self.last = self.d["a"][self.idx]
        self.idx += 1
        self.steps += 1
        done = (self.steps >= self.L) | (self.idx >= self.n - 1)
        selected = np.flatnonzero(done.cpu().numpy())
        if len(selected):
            self.reset(done)
            self._reset_events(selected)
        return self.observe(), latency, violation, done

    def state_dict(self):
        state = super().state_dict()
        state["events"] = {key: getattr(self, key) for key in self.event_state}
        return state

    def load_state_dict(self, state):
        state = dict(state)
        for key, value in state.pop("events").items():
            setattr(self, key, value)
        super().load_state_dict(state)
