import json
from PIL import Image
import os

from typing import Callable
from typing import Any
from torch.utils.data import Dataset
import torch
import numpy as np

from tqdm import tqdm
from scipy.optimize import linear_sum_assignment as linear_assignment



class Clevr4(Dataset):
    all_taxonomies = {
        'color': ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow", "pink", "orange"],
        'texture': ["rubber", "metal", "checkered", "emojis", "wave", "brick", "star", "circles", "zigzag",
                    "chessboard"],
        'count': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'shape': ["cube", "sphere", "monkey", "cone", "torus", "star", "teapot", "diamond", "gear", "cylinder"]
    }

    def __init__(
            self,
            root: str,
            taxonomy: str,
            transform: Callable = None,
    ):
        super().__init__()

        # Clevr4 Metadata
        self.root = root
        self.image_root = os.path.join(root, 'images')
        self.taxonomy = taxonomy
        self.transform = transform

        # Load annotations
        annot_path = os.path.join(root, 'clevr_4_annots.json')
        with open(annot_path, 'r') as f:
            annots = json.load(f)
        self.annotations = annots
        self.class_name_to_label = {
            name: idx for idx, name in enumerate(self.all_taxonomies[taxonomy])
        }

        # List files
        self.filenames = sorted(
            [fname for fname, meta in annots.items()]
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):

        # Load image
        fname = self.filenames[index]
        img_path = os.path.join(self.image_root, f"{fname}.png")
        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        # Label to index
        class_name = self.annotations[fname][self.taxonomy]
        target = self.class_name_to_label[class_name]

        return image, target

    def _get_all_taxonomy_targets(self, taxonomy):

        class_name_to_label = {
            name: idx for idx, name in enumerate(self.all_taxonomies[taxonomy])
        }

        all_targets = []
        for fname in self.filenames:
            class_name = self.annotations[fname][taxonomy]
            target = class_name_to_label[class_name]
            all_targets.append(target)

        return all_targets


@torch.no_grad()
def extract_features(
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 128,
        num_workers=8,
        device: torch.device = torch.device("cuda")
) -> np.array:
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size, num_workers=num_workers
    )

    all_feats = []
    for batch_idx, (images, _) in enumerate(tqdm(loader)):
        images = images.to(device)
        feats = model(images)
        all_feats.append(feats.cpu())

    all_feats = torch.cat(all_feats, dim=0)
    all_feats = all_feats.numpy()

    return all_feats



def cluster_acc(y_true, y_pred, return_ind=False):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    if return_ind:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind, w
    else:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


