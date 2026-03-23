from __future__ import annotations


def collate_fn(batch):
    """Keep the toy path explicit for batch_size=1 debugging.

    The returned batch preserves per-sample shapes:
    - coords: [N, 3]
    - atom_types: [N]
    - radii: [N]
    - query_points: [Q, 3]
    """
    if not batch:
        raise ValueError("batch must not be empty")
    if len(batch) != 1:
        raise ValueError("toy collate_fn only supports batch_size=1")
    return batch[0]
