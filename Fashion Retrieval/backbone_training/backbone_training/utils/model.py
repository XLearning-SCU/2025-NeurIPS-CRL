import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.hidden_dim = int((in_dim + out_dim) / 2)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, out_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class AttributeProjectionModel(nn.Module):
    def __init__(self, in_dim, out_dim, n_attr):
        super().__init__()
        self.n_attr = n_attr
        self.projection_heads = nn.ModuleList([MLP(in_dim, out_dim) for _ in range(n_attr)])

    def forward(self, x, attr_idx):
        assert torch.max(attr_idx) < self.n_attr, "Attribute index out of range"
        outputs = torch.zeros((x.size(0), self.projection_heads[0].fc[-1].out_features), device=x.device)
        for i in range(self.n_attr):
            mask = (attr_idx == i)
            if mask.any() and int(mask.sum())>1:
                outputs[mask] = self.projection_heads[i](x[mask])
        return outputs


def extract_features(model1, model2, data_loader, text, n_attrs, transform, device):
    feats = []
    indices = [[] for _ in range(n_attrs)]
    values = []
    with tqdm(total=len(data_loader)) as bar:
        cnt = 0
        for idx, (x, a, v) in enumerate(data_loader):
            a = a.to(device)
            x = torch.stack([transform(i) for i in x], dim=0).to(device)
            with torch.no_grad():
                x_img = model1.encode_image(x)
                x_sim = torch.bmm(x_img.unsqueeze(1), text[a]).squeeze(1)
                x_sim = x_sim / x_sim.norm(dim=-1, keepdim=True)
                new_x = model2(x_sim, a)
                feats.append(new_x.detach().cpu().numpy())
                values.append(v.numpy())
                for i in range(a.size(0)):
                    indices[a[i].cpu().item()].append(cnt)
                    cnt += 1
                bar.update(1)
    feats = np.concatenate(feats)
    feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    values = np.concatenate(values)
    feats = [feats[indices[i]] for i in range(n_attrs)]
    values = [values[indices[i]] for i in range(n_attrs)]
    return feats, values


























