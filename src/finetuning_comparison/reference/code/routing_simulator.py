"""Replay a common workload under oracle or completion-event observations."""

import numpy as np
from numba import njit


@njit(cache=True)
def survival(samples, usable_demand):
    return 1. - np.searchsorted(samples, usable_demand, side="right") / len(samples)


@njit(cache=True)
def remaining_demand(group, elapsed, bank, digits, prefix, means):
    lo, hi = digits[group], digits[group + 1]
    first = lo + np.searchsorted(bank[lo:hi], elapsed, side="right")
    if first == hi:

        return means[group]
    return max(0., (prefix[hi] - prefix[first]) / (hi - first) - elapsed)


@njit(cache=True)
def estimated_calendar(now, speeds, workers, heads, begins, ids,
                       pending, bank, offsets, prefix, means, bias, result):
    """Reveal starts and completions only when their event time has passed."""
    for j in range(len(workers)):
        residual = np.zeros(workers[j])
        for k in range(workers[j]):
            event = heads[j, k]
            if event >= 0 and begins[event] <= now:
                elapsed = (now - begins[event]) * speeds[j]
                residual[k] = remaining_demand(ids[event], elapsed, bank, offsets,
                                               prefix, means) / speeds[j]

        ordered = np.sort(residual)
        level = ordered[0]
        queued = max(0., pending[j]) / speeds[j]
        for k in range(1, workers[j]):
            fill = (ordered[k] - level) * k
            if queued < fill:
                level += queued / k
                queued = 0.
                break
            queued -= fill
            level = ordered[k]
        level += queued / workers[j]
        for k in range(workers[j]):
            result[j, k] = now + bias * max(level, residual[k])


@njit(cache=True)
def decide(now, last, p50, p99, pred, deadline, samples, avail, speeds, workers,
           scales, weights, method, guard, trigger, reduction):
    q = np.empty(len(speeds), np.float32)
    wait = np.empty(len(speeds))
    delay = np.empty(len(speeds))
    if method == 0:
        obs = np.empty(4 + 6 * len(speeds), np.float32)
        p50 = max(p50, 1e-6)
        p99 = max(p50, p99)
        obs[0] = min(1., max(0., np.log1p(p50) / scales[0]))
        obs[1] = min(1., max(0., np.log1p(p99) / scales[1]))
        obs[2] = min(1., max(0., p99 / p50 / scales[2]))
        obs[3] = min(1., max(0., np.log1p(max(0., now - last)) / scales[3]))
    for j in range(len(speeds)):
        rmin, rmax, total, busy = np.inf, 0., 0., 0.
        for k in range(workers[j]):
            r = max(0., avail[j, k] - now)
            rmin = min(rmin, r)
            rmax = max(rmax, r)
            total += r
            busy += r > 0
        wait[j] = rmin
        delay[j] = rmin + pred / speeds[j]
        if method == 0:
            pos = 4 + 6 * j
            obs[pos] = speeds[j] / np.max(speeds)
            obs[pos + 1] = workers[j] / np.max(workers)
            obs[pos + 2] = min(1., rmin / scales[4])
            obs[pos + 3] = min(1., total / workers[j] / scales[4])
            obs[pos + 4] = min(1., rmax / (2 * scales[4]))
            obs[pos + 5] = busy / workers[j]
    if method == 0:
        w0, b0, w1, b1, w2, b2 = weights
        h = np.maximum(np.dot(w0, obs) + b0, np.float32(0.))
        h = np.maximum(np.dot(w1, h) + b1, np.float32(0.))
        q = np.dot(w2, h) + b2
        proposal = np.argmax(q)
    else:
        proposal = np.argmin(delay)
    original_risk = survival(samples, (deadline - wait[proposal]) * speeds[proposal])
    pick, risk = proposal, original_risk
    if method == 2 or (guard and original_risk > trigger):
        for j in range(len(speeds)):
            if j == proposal:
                continue
            alternative = survival(samples, (deadline - wait[j]) * speeds[j])
            better = q[j] > q[pick] if method == 0 else delay[j] < delay[pick]
            if alternative < risk or (alternative == risk and better):
                pick, risk = j, alternative
    if method != 2 and original_risk - risk < reduction - 1e-15:
        pick, risk = proposal, original_risk
    return pick, proposal, risk, original_risk, wait[pick]


@njit(nogil=True, cache=True)
def replay(x, ids, preds, speeds, workers, scales, weights, method, guard, estimated,
           bias, bank, offsets, prefix, means, trigger=.005, reduction=.005):
    n, m, width = len(x), len(workers), np.max(workers)
    truth = np.full((m, width), np.inf)
    head = np.full((m, width), -1)
    tail = np.full((m, width), -1)
    links = np.full(n, -1)
    starts = np.empty(n)
    ends = np.empty(n)
    pending = np.zeros(m)
    predicted = truth.copy()
    for j in range(m):
        truth[j, :workers[j]] = x[0, 0]
    out = np.empty((n, 10))
    last = x[0, 0]
    for i in range(n):
        now = x[i, 0]
        avail = truth
        if estimated:

            for j in range(m):
                for k in range(workers[j]):
                    event = head[j, k]
                    while event >= 0 and starts[event] <= now:
                        if out[event, 9] == 1.:
                            pending[j] -= means[ids[event]]
                            out[event, 9] = 0.
                        if ends[event] > now:
                            break
                        event = links[event]
                    head[j, k] = event
            estimated_calendar(now, speeds, workers, head, starts, ids,
                               pending, bank, offsets, prefix, means, bias, predicted)
            avail = predicted
        group = ids[i]
        samples = bank[offsets[group]:offsets[group + 1]]
        action, proposal, risk, proposal_risk, expected = decide(
            now, last, x[i, 1], x[i, 2], preds[i], x[i, 4], samples, avail,
            speeds, workers, scales, weights, method, guard, trigger, reduction)

        k = np.argmin(truth[action, :workers[action]])
        start = max(now, truth[action, k])
        end = start + x[i, 3] / speeds[action]
        hypothetical_wait = max(0., np.min(truth[proposal, :workers[proposal]]) - now)
        out[i, 0] = end - now
        out[i, 1] = action
        out[i, 2] = proposal
        out[i, 3] = risk
        out[i, 4] = proposal_risk
        out[i, 5] = expected - (start - now)
        out[i, 6] = hypothetical_wait + x[i, 3] / speeds[proposal] > x[i, 4]
        out[i, 7] = start - now
        out[i, 8] = survival(samples, (x[i, 4] - (start - now)) * speeds[action])
        out[i, 9] = 0.
        truth[action, k] = end
        if estimated:
            starts[i], ends[i] = start, end

            if head[action, k] < 0:
                head[action, k] = i
            else:
                links[tail[action, k]] = i
            tail[action, k] = i
            if start > now:
                pending[action] += means[group]
                out[i, 9] = 1.
        last = now
    return out
