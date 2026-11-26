"""Historical per-function empirical demand samples."""

import numpy as np
import pandas as pd


def _guard_service_bank(path):
    train = pd.read_csv(path, usecols=['func_id', 'service_demand'], dtype={'func_id': str})
    if not np.isfinite(train.service_demand).all() or (train.service_demand < 0).any():
        raise ValueError('Invalid training service demand')
    names, offsets, blocks = {}, [0], []
    for name, group in train.groupby('func_id', sort=True):
        names[name] = len(blocks)
        work = np.sort(group.service_demand.to_numpy(np.float64))
        blocks.append(work)
        offsets.append(offsets[-1] + len(work))
    return names, np.concatenate(blocks), np.array(offsets, dtype=np.int64)
