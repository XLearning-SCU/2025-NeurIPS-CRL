import os
import torch
import random
from PIL import Image
from torch.utils.data import Dataset


def _image_reader(path):
    return Image.open(path).convert('RGB')


class BaseDataSet(Dataset):
	def __init__(self, cfg, split, transform=None):
		self.root_path = os.path.join(cfg.DATA.BASE_PATH, cfg.DATA.DATASET)

		self.fnamelist = []
		filepath = os.path.join(self.root_path, cfg.DATA.PATH_FILE[split])
		assert os.path.exists(filepath), f"File {filepath} does not exist."
		with open(filepath, 'r') as f:
			for l in f:
				self.fnamelist.append(l.strip())
		
		self.image_loader = _image_reader
		self.transform = transform

	def __len__(self):
		return self.fnamelist

	def __getitem__(self, index):
		path = os.path.join(self.root_path, self.fnamelist[index[0]])
		assert os.path.exists(path), f"File {path} does not exist."

		img = self.image_loader(path)
		img = self.transform(img)

		return (img,) + index[1:]


def triplet_collate_fn(batch):
	n = len(batch) // 3
	x, x_a = zip(*batch[:n])
	p, p_a = zip(*batch[n:2*n])
	n, n_a = zip(*batch[2*n:3*n])

	return x, p, n, torch.LongTensor(x_a)


def image_collate_fn(batch):
	x, a, v = zip(*batch)

	a = torch.LongTensor(a)
	v = torch.LongTensor(v)

	return x, a, v