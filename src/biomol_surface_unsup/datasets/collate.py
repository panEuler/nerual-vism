from __future__ import annotations


def collate_fn(batch):
    """Keep the toy path explicit.

    The returned batch preserves per-sample shapes:
    - coords: [N, 3]
    - atom_types: [N]
    - radii: [N]
    - query_points: [Q, 3]
    """
    if not batch:
        raise ValueError("batch must not be empty")
    return batch[0]
