from torchvision import transforms
import numpy as np
import cv2

import clip
import matplotlib.pyplot as plt

# _, preprocess = clip.load("RN50")
_, preprocess = clip.load("ViT-B/16")


class GlobalTransform(object):
	def __init__(self, cfg, is_train=False):
		self.t = transforms.Compose(
			[
				transforms.Resize(cfg.INPUT.GLOBAL_SIZE),
				transforms.CenterCrop(cfg.INPUT.GLOBAL_SIZE),
				transforms.ToTensor(),
				transforms.Normalize(
					mean=cfg.INPUT.PIXEL_MEAN,
					std=cfg.INPUT.PIXEL_STD
				)
			]
		)

	def __call__(self, img):
		return self.t(img)


class LocalTransform(object):
	def __init__(self, cfg):
		self.t = transforms.Compose(
			[
				# transforms.Resize(cfg.INPUT.LOCAL_SIZE),
				transforms.Resize(224),
				transforms.ToTensor(),
				transforms.Normalize(
					mean=cfg.INPUT.PIXEL_MEAN,
					std=cfg.INPUT.PIXEL_STD
				)
			]
		)

		self.global_size = cfg.INPUT.GLOBAL_SIZE
		self.threshold = cfg.INPUT.THRESHOLD

	def __call__(self, img, mask):
		# resize to global_size * global_size
		mask = cv2.resize(mask, (self.global_size,self.global_size), interpolation=cv2.INTER_LINEAR)
		# min-max normalization
		mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
		# map to char
		mask = np.uint8(mask * 255)
		# binarization
		ret, mask = cv2.threshold(mask, 255 * self.threshold, 255, cv2.THRESH_BINARY)

		# bounding box, got (left, top, width, height)
		x, y, w, h = cv2.boundingRect(mask)
		# box center
		xc = x + w//2
		yc = y + h//2
		# align to larger edge
		d = max(w, h)
		# handle case that patch out of image border
		if xc + d//2 > self.global_size:
			x = self.global_size - d
		else:
			x = max(0, xc - d//2)
		if yc + d//2 > self.global_size:
			y = self.global_size - d
		else:
			y = max(0, yc - d//2)

		# short edge is width or height
		short_edge = 0 if img.size[0] < img.size[1] else 1
		x = int(x / self.global_size * img.size[short_edge])
		y = int(y / self.global_size * img.size[short_edge])
		d = int(d / self.global_size * img.size[short_edge])
		if short_edge == 0:
			y += (img.size[1] - img.size[0]) // 2
		else:
			x += (img.size[0] - img.size[1]) // 2

		# origin_img = preprocess(img)
		# origin_img = origin_img.permute(1, 2, 0)
		# origin_img = (origin_img - origin_img.min()) / (origin_img.max() - origin_img.min())
		# origin_img = origin_img.detach().cpu().numpy()
		# plt.imsave('origin_img.png', origin_img)

		img = img.crop((x, y, x+d, y+d))

		# img = self.t(img)
		img = preprocess(img)

		# img = img.permute(1, 2, 0)
		# img = (img - img.min()) / (img.max() - img.min())
		# img = img.detach().cpu().numpy()
		# plt.imsave('crop_img.png', img)
		# exit(0)

		return img
