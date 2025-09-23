# Copyright (c) Meta Platforms, Inc. and affiliates.
import numpy as np
import torch
from gen_utils import AverageMeter
from metric import get_recall
from tqdm import tqdm


@torch.no_grad()
def validate(clip_model, combiner, valloader, text_embedding, topk=(1, 2, 3)):
    print('Computing eval with combiner...')

    clip_model.eval()
    combiner.eval()

    meters = {k: AverageMeter() for k in topk}

    with torch.no_grad():
        for ref_img, gallery_set, target_rank, caption_text, caption_token in tqdm(valloader):

            ref_img, gallery_set, target_rank, caption_token = ref_img.cuda(), gallery_set.cuda(), target_rank.cuda(), caption_token.cuda()
            bsz, n_gallery, _, h, w = gallery_set.size()

            caption_token = caption_token.squeeze(dim=0)
            # print(caption_token.shape)

            # Forward pass in CLIP
            imgs_ = torch.cat([ref_img, gallery_set.view(-1, 3, h, w)], dim=0)
            all_img_feats = clip_model.encode_image(imgs_).float()
            caption_feats = clip_model.encode_text(caption_token).float()
            caption_feats = torch.nn.functional.normalize(caption_feats, dim=-1)

            # L2 normalize and view into correct shapes
            ref_feats, gallery_feats = all_img_feats.split((bsz, bsz * n_gallery), dim=0)
            gallery_feats = gallery_feats.view(bsz, n_gallery, -1)
            gallery_feats = torch.nn.functional.normalize(gallery_feats, dim=-1)

            # Forward pass in combiner
            combined_feats = combiner(ref_feats, caption_feats)

            # Compute similarity
            img_sim = combined_feats[:, None, :] * gallery_feats  # B x N x D
            img_sim = img_sim.sum(dim=-1)  # B x N

            # new_ref = combined_feats @ traits.T
            new_ref = ref_feats @ text_embedding.T
            new_ref = torch.nn.functional.normalize(new_ref, dim=-1)
            # new_ref = new_ref - torch.mean(new_ref)
            # new_ref /= torch.std(new_ref, dim=-1 ,keepdim=True)


            gallery_feats = gallery_feats.squeeze(dim=0)
            new_gallery = gallery_feats @ text_embedding.T
            new_gallery = torch.nn.functional.normalize(new_gallery, dim=-1)
            # new_gallery -= torch.mean(new_gallery)
            # new_gallery /= torch.std(new_gallery, dim=-1 ,keepdim=True)



            new_sim = new_ref @ new_gallery.T
            new_sim = torch.nn.functional.normalize(new_sim, dim=-1)

            condition_sim = caption_feats @ gallery_feats.T
            if condition_sim.shape[0] > 1:
                condition_sim = condition_sim.mean(dim=0, keepdim=True)
            condition_sim = torch.nn.functional.normalize(condition_sim, dim=-1)


            sim = new_sim * 10 + condition_sim

            _, sort_idxs = sim.sort(dim=-1, descending=True)  # B x N

            # Compute recall at K
            for k in topk:
                recall_k = get_recall(sort_idxs[:, :k], target_rank)
                meters[k].update(recall_k, bsz)


        # Print results
        print_str = '\n'.join([f'Recall @ {k} = {v.avg:.4f}' for k, v in meters.items()])
        print(print_str)

        return meters

