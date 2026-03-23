import torch
import torch.nn as nn
from .atom_features import AtomFeatureEmbedding

class GlobalFeatureEncoder(nn.Module):
    def __init__(self, num_atom_types: int, atom_embed_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.atom_embedding = AtomFeatureEmbedding(num_atom_types, atom_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(3 + atom_embed_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, coords, atom_types, radii):
        atom_emb = self.atom_embedding(atom_types)
        x = torch.cat([coords, atom_emb, radii.unsqueeze(-1)], dim=-1)
        h = self.mlp(x)
        pooled = h.mean(dim=0)
        return self.out(pooled)