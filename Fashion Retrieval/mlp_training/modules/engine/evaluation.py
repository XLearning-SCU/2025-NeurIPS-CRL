import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from modules.utils.metric import APScorer, AverageMeter


def do_eval(
	clip_model,
	model,
	query_loader,
	candidate_loader,
	gt,
	lt,
	attrs,
	device,
	logger,
	epoch=-1,
	beta=0.6
):
	mAPs = AverageMeter()

	logger.info("Begin evaluation.")
	model.eval()

	logger.info("Forwarding query images...")
	q_feats, q_values = extract_features(clip_model, model, query_loader, gt, lt, device, len(attrs), beta=beta)
	logger.info("Forwarding candidate images...")
	c_feats, c_values = extract_features(clip_model, model, candidate_loader, gt, lt, device, len(attrs), beta=beta)

	# return q_feats, q_values, c_feats, c_values


	for i, attr in enumerate(attrs):
		mAP = mean_average_precision(q_feats[i], c_feats[i], q_values[i], c_values[i])
		logger.info(f"{attr} MeanAP: {100.*mAP:.4f}")
		mAPs.update(mAP, q_feats[i].shape[0])

	logger.info(f"Total MeanAP: {100.*mAPs.avg:.4f}")

	return mAPs.avg


def extract_features(clip_model, model, data_loader, gt, lt, device, n_attrs, beta=0.6):
	feats = []
	indices = [[] for _ in range(n_attrs)]
	values = []
	with tqdm(total=len(data_loader)) as bar:
		cnt = 0
		for idx, batch in enumerate(data_loader):
			x, a, v = batch
			a = a.to(device)

			out = process_batch(clip_model, model, x, a, gt, lt, device, beta=beta)
			feats.append(out.detach().cpu().numpy())
			values.append(v.numpy())

			for i in range(a.size(0)):
				indices[a[i].cpu().item()].append(cnt)
				cnt += 1

			bar.update(1)

	feats = np.concatenate(feats)
	values = np.concatenate(values)

	feats = [feats[indices[i]] for i in range(n_attrs)]
	values = [values[indices[i]] for i in range(n_attrs)]

	return feats, values


def process_batch(clip_model, model, x, a, gt, lt, device, beta=0.6):
	gx = torch.stack([gt(i) for i in x], dim=0)
	gx = gx.to(device)
	with torch.no_grad():
		g_feats, attmap = model(gx, a, level='global')

	if lt is None:
		return nn.functional.normalize(g_feats, p=2, dim=1)

	attmap = attmap.cpu().numpy()
	lx = torch.stack([lt(i, mask) for i, mask in zip(x, attmap)], dim=0)
	lx = lx.to(device)

	out = clip_model.encode_image(lx)

	# lx = lx.to(device)
	# with torch.no_grad():
	# 	l_feats, _ = model(lx, a, level='local')
	#
	# out = torch.cat((torch.sqrt(torch.tensor(beta)) * nn.functional.normalize(g_feats, p=2, dim=1),
	# 		torch.sqrt(torch.tensor(1-beta)) * nn.functional.normalize(l_feats, p=2, dim=1)), dim=1)

	return out


def mean_average_precision(queries, candidates, q_values, c_values):
    '''
    calculate mAP of a conditional set. Samples in candidate and query set are of the same condition.
        cand_set: 
            type:   nparray
            shape:  c x feature dimension
        queries:
            type:   nparray
            shape:  q x feature dimension
        c_gdtruth:
            type:   nparray
            shape:  c
        q_gdtruth:
            type:   nparray
            shape:  q
    '''
 
    scorer = APScorer(candidates.shape[0])

    # similarity matrix
    simmat = np.matmul(queries, candidates.T)

    ap_sum = 0
    for q in range(simmat.shape[0]):
        sim = simmat[q]
        index = np.argsort(-sim)
        sorted_labels = []
        for i in range(index.shape[0]):
            if c_values[index[i]] == q_values[q]:
                sorted_labels.append(1)
            else:
                sorted_labels.append(0)
        
        ap = scorer.score(sorted_labels)
        ap_sum += ap

    mAP = ap_sum / simmat.shape[0]

    return mAP