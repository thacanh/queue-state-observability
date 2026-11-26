"""Add queue diagnostics and two observation interventions to the frozen replay."""

import inspect

import numpy as np
from numba import njit

import routing_simulator
from crossed_router import change
from routing_simulator import estimated_calendar, survival


@njit
def perturb(now, truth, workers, noise, bias, sigma, output):
    for j in range(len(workers)):
        for k in range(workers[j]):
            residual = max(0., truth[j, k] - now)
            output[j, k] = (now + max(0., residual + bias + sigma * noise[j, k])
                            if residual > 0. else truth[j, k])


@njit
def residual_error(now, view, truth, workers):
    total, absolute, count = 0., 0., 0
    for j in range(len(workers)):
        for k in range(workers[j]):
            error = max(0., view[j, k] - now) - max(0., truth[j, k] - now)
            total += error
            absolute += abs(error)
            count += 1
    return total / count, absolute / count


def make_kernel():
    decide = inspect.getsource(routing_simulator.decide.py_func).split("\n", 1)[1]
    decide = change(decide, "trigger, reduction):", "trigger, reduction, risk_avail):")
    decide = change(decide, "wait[j] = rmin", "wait[j] = max(0., np.min(risk_avail[j, :workers[j]]) - now)")
    decide = change(decide, "return pick, proposal, risk, original_risk, wait[pick]",
        """upper = np.zeros(7)
    lower = np.zeros(7)
    if method == 0:
        for k in range(4):
            upper[k] = obs[k] >= 1.
            lower[k] = obs[k] <= 0.
        for j in range(len(speeds)):
            for k in range(3):
                v = obs[4+6*j+2+k]
                upper[4+k] += (v >= 1.) / len(speeds)
                lower[4+k] += (v <= 0.) / len(speeds)
    return pick, proposal, risk, original_risk, wait[pick], upper, lower""")
    scope = {"np": np, "survival": survival, "estimated_calendar": estimated_calendar,
             "perturb": perturb, "residual_error": residual_error}
    exec(compile(decide, "<diagnostic decision>", "exec"), scope)
    scope["decide"] = njit(scope["decide"])
    replay = inspect.getsource(routing_simulator.replay.py_func).split("\n", 1)[1]
    replay = change(replay, "bias, bank, offsets, prefix, means, trigger=.005, reduction=.005):",
                    "bias, bank, offsets, prefix, means, trigger, reduction, channel, error, sigma, noise):")
    replay = change(replay, "    predicted = truth.copy()", """    predicted = truth.copy()
    altered = truth.copy()
    queued = np.zeros(m, dtype=np.int64)
    pending_real = np.zeros(m)""")
    replay = change(replay, "np.empty((n, 10))", "np.empty((n, 31 + 7*m))")
    replay = change(replay, "                            pending[j] -= means[ids[event]]",
                    """                            pending[j] -= means[ids[event]]
                            queued[j] -= 1
                            pending_real[j] -= x[event, 3]""")
    replay = change(replay, "        group = ids[i]", """        risk_view = avail
        if channel != 0:
            perturb(now, truth, workers, noise[i], error, sigma, altered)
            if channel == 1:
                avail = altered
            else:
                risk_view = altered
        group = ids[i]""")
    replay = change(replay, "proposal_risk, expected = decide(", "proposal_risk, expected, upper, lower = decide(")
    replay = change(replay, "speeds, workers, scales, weights, method, guard, trigger, reduction)",
                    "speeds, workers, scales, weights, method, guard, trigger, reduction, risk_view)")
    replay = change(replay, "        truth[action, k] = end", """        out[i, 10:17] = upper
        out[i, 17:24] = lower
        out[i, 24] = start
        out[i, 25] = end
        out[i, 26] = guard and proposal_risk > trigger
        out[i, 27], out[i, 28] = residual_error(now, avail, truth, workers)
        out[i, 29], out[i, 30] = residual_error(now, risk_view, truth, workers)
        for j in range(m):
            pos = 31 + 7*j
            r = np.maximum(0., avail[j, :workers[j]] - now)
            out[i, pos] = np.min(r)
            out[i, pos+1] = np.mean(r)
            out[i, pos+2] = np.max(r)
            out[i, pos+3] = max(0., np.min(truth[j, :workers[j]]) - now)
            out[i, pos+4] = queued[j]
            out[i, pos+5] = max(0., pending[j])
            out[i, pos+6] = max(0., pending_real[j])
        truth[action, k] = end""")
    replay = change(replay, "                pending[action] += means[group]", """                pending[action] += means[group]
                queued[action] += 1
                pending_real[action] += x[i, 3]""")
    exec(compile(replay, "<diagnostic replay>", "exec"), scope)
    return njit(nogil=True)(scope["replay"])
