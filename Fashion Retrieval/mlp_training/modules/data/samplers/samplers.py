import os
import math
import random
from torch.utils.data.sampler import Sampler


class TripletSampler(Sampler):
	def __init__(self, cfg):
		self.num_triplets = cfg.DATA.NUM_TRIPLETS
		self.batch_size = cfg.DATA.TRAIN_BATCHSIZE
		self.attrs = cfg.DATA.ATTRIBUTES.NAME
		self.num_values = cfg.DATA.ATTRIBUTES.NUM
		self.indices = {}
		for i, attr in enumerate(self.attrs):
			self.indices[attr] = [[] for _ in range(self.num_values[i])]

		label_file = os.path.join(cfg.DATA.BASE_PATH, cfg.DATA.DATASET, cfg.DATA.GROUNDTRUTH.TRAIN)
		assert os.path.exists(label_file), f"Train label file {label_file} does not exist."
		with open(label_file, 'r') as f:
			for l in f:
				l = [int(i) for i in l.strip().split()]
				fid = l[0]
				attr_val = [(l[i], l[i+1]) for i in range(1, len(l), 2)]
				for attr, val in attr_val:
					self.indices[self.attrs[attr]][val].append(fid)


	def __len__(self):
		return math.ceil(self.num_triplets / self.batch_size)

	def __str__(self):
		return f"| Triplet Sampler | iters {self.__len__()} | batch size {self.batch_size}|"

	def __iter__(self):
		sampled_attrs = random.choices(range(0, len(self.attrs)), k=self.num_triplets)
		for i in range(self.__len__()):
			attrs = sampled_attrs[i*self.batch_size:(i+1)*self.batch_size]

			anchors = []
			positives = []
			negatives = []
			for a in attrs:
				# Randomly sample two attribute values
				vp, vn = random.sample(range(self.num_values[a]), 2)
				# Randomly sample an anchor image and a positive image
				x, p = random.sample(self.indices[self.attrs[a]][vp], 2)
				# Randomly sample a negative image
				n = random.choice(self.indices[self.attrs[a]][vn])
				anchors.append((x, a))
				positives.append((p, a))
				negatives.append((n, a))

			yield anchors + positives + negatives


class ImageSampler(Sampler):
	def __init__(self, cfg, file):
		self.batch_size = cfg.DATA.TEST_BATCHSIZE

		label_file = os.path.join(cfg.DATA.BASE_PATH, cfg.DATA.DATASET, file)
		assert os.path.exists(label_file), f"Train label file {label_file} does not exist."
		self.labels = []
		with open(label_file, 'r') as f:
			for l in f:
				l = [int(i) for i in l.strip().split()]
				self.labels.append(tuple(l))

	def __len__(self):
		return math.ceil(len(self.labels) / self.batch_size)

	def __str__(self):
		return f"| Image Sampler | iters {self.__len__()} | batch size {self.batch_size}|"

	def __iter__(self):
		for i in range(self.__len__()):
			yield self.labels[i*self.batch_size:(i+1)*self.batch_size]