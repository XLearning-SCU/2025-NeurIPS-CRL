from .model import ASEN


def build_model(cfg):
	return ASEN(cfg)