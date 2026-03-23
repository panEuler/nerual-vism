def train_step(model, batch, loss_fn, optimizer, device):
    model.train()
    coords = batch["coords"].to(device)
    atom_types = batch["atom_types"].to(device)
    radii = batch["radii"].to(device)
    query_points = batch["query_points"].to(device).requires_grad_(True)

    out = model(coords, atom_types, radii, query_points)
    losses = loss_fn(
        {"coords": coords, "atom_types": atom_types, "radii": radii, "query_points": query_points},
        out,
    )
    optimizer.zero_grad()
    losses["total"].backward()
    optimizer.step()
    return {k: float(v.detach().cpu()) for k, v in losses.items()}