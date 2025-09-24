from torch.utils.data import DataLoader
from .datasets import BaseDataSet, triplet_collate_fn, image_collate_fn
from .samplers import TripletSampler, ImageSampler


def build_data(cfg, transform, test_on=None):
	if test_on is None:
		train_set = BaseDataSet(cfg, 'TRAIN', transform)
		valid_set = BaseDataSet(cfg, 'VALID', transform)

		train_loader = DataLoader(
			train_set,
			collate_fn=triplet_collate_fn,
			batch_sampler=TripletSampler(cfg),
			num_workers=cfg.DATA.NUM_WORKERS,
			pin_memory=True
		)

		valid_query_loader = DataLoader(
			valid_set,
			collate_fn=image_collate_fn,
			batch_sampler=ImageSampler(cfg, cfg.DATA.GROUNDTRUTH.QUERY.VALID),
			num_workers=cfg.DATA.NUM_WORKERS,
			shuffle=False,
			pin_memory=True
		)

		valid_candidate_loader = DataLoader(
			valid_set,
			collate_fn=image_collate_fn,
			batch_sampler=ImageSampler(cfg, cfg.DATA.GROUNDTRUTH.CANDIDATE.VALID),
			num_workers=cfg.DATA.NUM_WORKERS,
			shuffle=False,
			pin_memory=True
		)

		return train_loader, valid_query_loader, valid_candidate_loader
	else:
		test_set = BaseDataSet(cfg, transform, test_on)

		test_query_loader = DataLoader(
			test_set,
			collate_fn=image_collate_fn,
			batch_sampler=ImageSampler(cfg, cfg.DATA.GROUNDTRUTH.QUERY[test_on]),
			num_workers=cfg.DATA.NUM_WORKERS,
			shuffle=False,
			pin_memory=True
		)

		test_candidate_loader = DataLoader(
			test_set,
			collate_fn=image_collate_fn,
			batch_sampler=ImageSampler(cfg, cfg.DATA.GROUNDTRUTH.CANDIDATE[test_on]),
			num_workers=cfg.DATA.NUM_WORKERS,
			shuffle=False,
			pin_memory=True
		)

		return test_query_loader, test_candidate_loader