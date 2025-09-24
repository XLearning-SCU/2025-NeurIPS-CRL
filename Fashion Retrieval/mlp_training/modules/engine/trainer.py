import time
import torch
import torch.nn as nn

from modules.utils.metric import AverageMeter


def do_train(cfg, model, data_loader, gt, lt, optimizer, criterion, device, logger, epoch):
	losses = AverageMeter()
	if lt is not None:
		glosses = AverageMeter()
		llosses = AverageMeter()
		alosses = AverageMeter()
	batch_time = AverageMeter()
	data_time = AverageMeter()

	model.train()

	end = time.time()
	for idx, batch in enumerate(data_loader):
		x, p, n, a = batch
		n_data = len(x)
		a = a.to(device)

		gx = torch.stack([gt(i) for i in x], dim=0).to(device)
		gp = torch.stack([gt(i) for i in p], dim=0).to(device)
		gn = torch.stack([gt(i) for i in n], dim=0).to(device)

		data_time.update(time.time() - end)

		gx, gx_attnmap = model(gx, a, level='global')
		gp, gp_attnmap = model(gp, a, level='global')
		gn, gn_attnmap = model(gn, a, level='global')

		loss = cfg.SOLVER.GLOBAL_WEIGHT * criterion(gx, gp, gn)

		if lt is not None:
			glosses.update(loss.cpu().item(), n_data)

			gx_attnmap = gx_attnmap.cpu().detach().numpy()
			gp_attnmap = gp_attnmap.cpu().detach().numpy()
			gn_attnmap = gn_attnmap.cpu().detach().numpy()
			lx = torch.stack([lt(i, mask) for i, mask in zip(x, gx_attnmap)], dim=0).to(device)
			lp = torch.stack([lt(i, mask) for i, mask in zip(p, gp_attnmap)], dim=0).to(device)
			ln = torch.stack([lt(i, mask) for i, mask in zip(n, gn_attnmap)], dim=0).to(device)

			lx, _ = model(lx, a, level='local')
			lp, _ = model(lp, a, level='local')
			ln, _ = model(ln, a, level='local')

			# local losses
			l = local_loss(criterion, gx, gp, gn, lx, lp, ln)
			llosses.update(cfg.SOLVER.LOCAL_WEIGHT * l[0].cpu().item(), n_data)
			alosses.update(cfg.SOLVER.ALIGN_WEIGHT * l[1].cpu().item(), n_data)
			loss += cfg.SOLVER.LOCAL_WEIGHT * l[0] + cfg.SOLVER.ALIGN_WEIGHT * l[1]

		losses.update(loss.cpu().item(), n_data)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		batch_time.update(time.time() - end)
		end = time.time()


		local_log = (f"Global Loss: {glosses.val:.4f}({glosses.avg:.4f})\t"+\
					f"Local Loss: {llosses.val:.4f}({llosses.avg:.4f})\t"+\
					f"Align Loss: {alosses.val:.4f}({alosses.avg:.4f})\t") if lt is not None else ""
		if idx % cfg.SOLVER.LOG_PERIOD == 0:
			logger.info(f"Train Epoch: [{epoch}][{idx}/{len(data_loader)}]\t"+
						local_log+
			 			f"Loss: {losses.val:.4f}({losses.avg:.4f})\t"+
			 			f"Batch Time: {batch_time.val:.3f}({batch_time.avg:.3f})\t"+
			 			f"Data Time: {data_time.val:.3f}({data_time.avg:.3f})")



def local_loss(criterion, gx, gp, gn, lx, lp, ln):
	lt_loss = criterion(lx, lp, ln)
	sim_x_ins = nn.functional.cosine_similarity(gx, lx, dim=1)
	sim_p_ins = nn.functional.cosine_similarity(gp, lp, dim=1)
	sim_n_ins = nn.functional.cosine_similarity(gn, ln, dim=1)
	a_loss = torch.mean(torch.clamp(1.-sim_x_ins, min=0)) + \
			 torch.mean(torch.clamp(1.-sim_p_ins, min=0)) + \
			 torch.mean(torch.clamp(1.-sim_n_ins, min=0))

	return lt_loss, a_loss
