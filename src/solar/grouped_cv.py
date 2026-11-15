"""Deterministic, metadata-only date-grouped cross-validation manifests."""
import hashlib
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def observation_date(name: str) -> str:
    """Read and validate the observation date encoded in an image name."""
    date = Path(name).name[:8]
    datetime.strptime(date, '%Y%m%d')
    return date


def make_grouped_folds(pool: list[str], holdout: list[str], seed: int,
                       n_splits: int = 5) -> list[dict]:
    """Balance image counts, using year counts then seeded hashes for ties."""
    if n_splits < 2:
        raise ValueError('At least two folds required')
    if len(pool) != len(set(pool)) or len(holdout) != len(set(holdout)):
        raise ValueError('Duplicate image names')
    if set(pool) & set(holdout):
        raise ValueError('Pool/holdout image overlap')
    groups = defaultdict(list)
    for name in sorted(pool):
        groups[observation_date(name)].append(name)
    if set(groups) & {observation_date(n) for n in holdout}:
        raise ValueError('Pool/holdout date overlap')
    if len(groups) < n_splits:
        raise ValueError('Insufficient distinct dates')

    def key(value: str) -> str:
        return hashlib.sha256(f'{seed}|{value}'.encode('utf-8')).hexdigest()

    bins: list[list[str]] = [[] for _ in range(n_splits)]
    years = [Counter() for _ in bins]
    dates = sorted(groups, key=lambda d: (-len(groups[d]), key('date|' + d), d))
    for date in dates:
        index = min(range(n_splits), key=lambda i: (
            len(bins[i]), years[i][date[:4]], key(f'fold|{date}|{i}'), i))
        bins[index].extend(groups[date])
        years[index][date[:4]] += len(groups[date])
    return [{'fold_id': i + 1, 'fit': sorted(set(pool) - set(names)),
             'development': sorted(names)} for i, names in enumerate(bins)]
