import torch
import torch.nn as nn

from .resnet import get_model


class ASEN(nn.Module):
	def __init__(self, cfg):
		super(ASEN, self).__init__()
		self.choices = nn.ModuleDict({
			'global': nn.ModuleDict({
				'attrnet': AttrEmbedding(cfg.DATA.NUM_ATTRIBUTES, cfg.MODEL.ATTRIBUTE.EMBED_SIZE),
				'basenet': get_model(cfg.MODEL.GLOBAL.BACKBONE.NAME, pretrained=True),
				'attnnet': AttnEmbedding(
					cfg.MODEL.ATTRIBUTE.EMBED_SIZE,
					cfg.MODEL.GLOBAL.BACKBONE.EMBED_SIZE,
					cfg.MODEL.GLOBAL.ATTENTION.SPATIAL.COMMON_EMBED_SIZE,
					cfg.MODEL.GLOBAL.ATTENTION.CHANNEL.REDUCTION_RATE,
					cfg.MODEL.EMBED_SIZE,
					cfg.MODEL.GLOBAL.ATTENTION.SPATIAL.ENABLE,
					cfg.MODEL.GLOBAL.ATTENTION.CHANNEL.ENABLE
				)
			})
		})

		if cfg.MODEL.LOCAL.ENABLE:
			self.choices.update(
				{'local': nn.ModuleDict({
					'attrnet': self.choices['global']['attrnet'],
					'basenet': get_model(cfg.MODEL.LOCAL.BACKBONE.NAME, pretrained=True),
					'attnnet': AttnEmbedding(
						cfg.MODEL.ATTRIBUTE.EMBED_SIZE,
						cfg.MODEL.LOCAL.BACKBONE.EMBED_SIZE,
						cfg.MODEL.LOCAL.ATTENTION.SPATIAL.COMMON_EMBED_SIZE,
						cfg.MODEL.LOCAL.ATTENTION.CHANNEL.REDUCTION_RATE,
						cfg.MODEL.EMBED_SIZE,
						cfg.MODEL.LOCAL.ATTENTION.SPATIAL.ENABLE,
						cfg.MODEL.LOCAL.ATTENTION.CHANNEL.ENABLE
					)
				})})

	def forward(self, x, a, level='global'):
		a = self.choices[level]['attrnet'](a)

		x = self.choices[level]['basenet'](x)
		x, attmap = self.choices[level]['attnnet'](x, a)


		return x, attmap

	def load_state_dict(self, loaded_state_dict):
		state = super(ASEN, self).state_dict()
		for k in loaded_state_dict:
			if k in state:
				state[k] = loaded_state_dict[k]
		super(ASEN, self).load_state_dict(state)


class AttrEmbedding(nn.Module):
	def __init__(self, n_attrs, embed_size):
		super(AttrEmbedding, self).__init__()
		self.attr_embedding  = torch.nn.Embedding(n_attrs, embed_size)

	def forward(self, x):
		return self.attr_embedding(x)


class AttnEmbedding(nn.Module):
	def __init__(
		self, 
		attr_embed_size, 
		img_embed_size, 
		common_embed_size, 
		reduction_rate, 
		embed_size,
		spatial_en=True,
		channel_en=True):
		super(AttnEmbedding, self).__init__()

		self.spatial_en = spatial_en
		self.channel_en = channel_en

		if self.spatial_en:
			self.attr_transform1 = nn.Linear(
				attr_embed_size, 
				common_embed_size
			)
			self.conv = nn.Conv2d(
				img_embed_size, 
				common_embed_size, 
				kernel_size=1, stride=1
			)

		if self.channel_en:
			self.attr_transform2 = nn.Linear(
				attr_embed_size, 
				attr_embed_size
			)
			self.fc1 = nn.Linear(
				img_embed_size+attr_embed_size,
				img_embed_size//reduction_rate
			)
			self.fc2 = nn.Linear(
				img_embed_size//reduction_rate,
				img_embed_size
			)

		self.feature_fc = nn.Linear(
			img_embed_size,
			embed_size
		)

		self.tanh = nn.Tanh()
		self.relu = nn.ReLU(inplace=True)
		self.softmax = nn.Softmax(dim=2)
		self.sigmoid = nn.Sigmoid()
		self.aapool = nn.AdaptiveAvgPool2d(1)

	def forward(self, x, a):
		if self.spatial_en:
			attmap = self.spatial_attn(x, a)
			x = x * attmap
			x = x.view(x.size(0), x.size(1), -1)
			x = x.sum(dim=2)
		else:
			x = self.aapool(x).squeeze()

		if self.channel_en:
			m = self.channel_attn(x, a)
			x = x * m

		x = self.feature_fc(x)

		return x, attmap.squeeze() if self.spatial_en else None

	def spatial_attn(self, x, a):
		x = self.conv(x)
		x = self.tanh(x)

		a = self.attr_transform1(a)
		a = self.tanh(a)
		a = a.view(a.size(0), a.size(1), 1, 1)
		a = a.expand_as(x)

		attmap = a * x
		attmap = torch.sum(attmap, dim=1, keepdim=True)
		attmap = torch.div(attmap, x.size(1) ** 0.5)
		attmap = attmap.view(attmap.size(0), attmap.size(1), -1)
		attmap = self.softmax(attmap)
		attmap = attmap.view(attmap.size(0), attmap.size(1), x.size(2), x.size(3))

		return attmap

	def channel_attn(self, x, a):
		a = self.attr_transform2(a)
		a = self.relu(a)

		cnt = torch.cat((x, a), dim=1)
		m = self.fc1(cnt)
		m = self.relu(m)
		m = self.fc2(m)
		m = self.sigmoid(m)

		return m