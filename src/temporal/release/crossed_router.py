"""Allow the routing policy and the guard to use separate queue observations."""

import inspect

import numpy as np
from numba import njit

import routing_simulator
from routing_simulator import estimated_calendar, survival


def change(source, old, new, count=1):
    if source.count(old) != count:
        raise ValueError(f"Routing simulator source anchor changed: {old}")
    return source.replace(old, new)


def crossed_kernel():
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
    scope = {"np": np, "survival": survival, "estimated_calendar": estimated_calendar}
    exec(compile(decide, "<crossed observation decision>", "exec"), scope)
    scope["decide"] = njit(scope["decide"])
    replay = inspect.getsource(routing_simulator.replay.py_func).split("\n", 1)[1]
    replay = change(replay, "bias, bank, offsets, prefix, means, trigger=",
                    "bias, bank, offsets, prefix, means, guard_estimated, trigger=")
    replay = change(replay, "if estimated:", "if estimated or guard_estimated:", count=2)
    replay = change(replay, "            avail = predicted",
                    "            if estimated:\n                avail = predicted")
    replay = change(replay, "np.empty((n, 10))", "np.empty((n, 24))")
    replay = change(replay, "proposal_risk, expected = decide(",
                    "proposal_risk, expected, upper, lower = decide(")
    replay = change(replay, "avail,\n            speeds, workers, scales, weights, method, guard, trigger, reduction)",
                    "avail,\n            speeds, workers, scales, weights, method, guard, trigger, reduction,\n"
                    "            predicted if guard_estimated else truth)")
    replay = change(replay, "out[i, 9] = 0.", "out[i, 9] = 0.\n        out[i, 10:17] = upper\n        out[i, 17:24] = lower")
    exec(compile(replay, "<crossed observation replay>", "exec"), scope)
    return njit(nogil=True)(scope["replay"])
