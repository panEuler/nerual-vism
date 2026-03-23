from .vism_lite import vism_lite_loss

def build_loss_fn(cfg):
    def loss_fn(batch, model_out):
        return vism_lite_loss(
            coords=batch["coords"],
            radii=batch["radii"],
            query_points=batch["query_points"],
            pred_sdf=model_out["sdf"],
        )
    return loss_fn