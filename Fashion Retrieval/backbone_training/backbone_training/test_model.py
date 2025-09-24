import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import torch
import numpy as np
import clip
import argparse
import random
from config import cfg
from datetime import datetime
from data.build import build_data
from utils.model import AttributeProjectionModel, extract_features
from utils.loss import TripletRankingLoss
from utils.metric import mean_average_precision, AverageMeter
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR




def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", nargs="+", help="config file",
        default=['/xlearning/honglin/project/clip-asen/config/DeepFashion.yaml'], type=str
    )
    parser.add_argument(
        "--seed", default=0, type=int, help="random seed"
    )
    args = parser.parse_args()
    return args



if __name__ == "__main__":

    args = get_parse()

    if args.cfg is not None:
        for cfg_file in args.cfg:
            cfg.merge_from_file(cfg_file)
    cfg.freeze()

    device = cfg.DEVICE

    text_path = os.path.join("./data", cfg.DATA.DATASET, "")

    # text_embedding = np.load(os.path.join(text_path, "text_embedding.npy"))
    # text_embedding = torch.from_numpy(text_embedding).to(device).to(torch.float32)
    # print(f"text_embedding.shape: {text_embedding.shape}")
    # text_embedding = text_embedding.transpose(1, 2)
    #
    # gpt_embedding = np.load(os.path.join(text_path, "gpt_embedding.npy"))[:, :100, :]
    # gpt_embedding = torch.from_numpy(gpt_embedding).to(device).to(torch.float32)
    # print(f"gpt_embedding.shape: {gpt_embedding.shape}")
    # gpt_embedding = gpt_embedding.transpose(1, 2)

    common_embedding = np.load(os.path.join(text_path, "common_embedding.npy"))
    common_embedding = torch.from_numpy(common_embedding).to(device).to(torch.float32)
    print(f"common_embedding.shape: {common_embedding.shape}")
    common_embedding = common_embedding.transpose(1, 2)

    text_here = common_embedding

    clip_model, preprocess = clip.load('ViT-B/16', device)
    checkpoint_clip = torch.load(os.path.join("/xlearning/honglin/project/clip-asen/data/DeepFashion", "100_only_backbone.pth"), map_location=device, weights_only=False)
    clip_model.visual.load_state_dict(checkpoint_clip['clip_visual'])
    clip_model = clip_model.to(torch.float32)

    valid_best_map, valid_best_epoch = checkpoint_clip['mAP'], checkpoint_clip['epoch']
    print(f"Validation best map: {100 * valid_best_map: .4f}, epoch: {valid_best_epoch}")


    mlp_model = AttributeProjectionModel(in_dim=text_here.shape[2], out_dim=1024, n_attr=cfg.DATA.NUM_ATTRIBUTES).to(device)
    checkpoint_mlp = torch.load(os.path.join("/xlearning/honglin/project/asen++/data/DeepFashion", "2025-01-21-15:32_best_model.pth"), map_location=device, weights_only=False)
    mlp_model.load_state_dict(checkpoint_mlp['model'])
    mlp_model = mlp_model.to(torch.float32)

    test_query_loader, test_candidate_loader = build_data(cfg, 'TEST')

    clip_model.visual.eval()
    mlp_model.eval()

    print("Forwarding query images...")
    q_feats, q_values = extract_features(clip_model, mlp_model, test_query_loader, text_here,
                                         cfg.DATA.NUM_ATTRIBUTES, preprocess, device)
    print("Forwarding candidate images...")
    c_feats, c_values = extract_features(clip_model, mlp_model, test_candidate_loader, text_here,
                                         cfg.DATA.NUM_ATTRIBUTES, preprocess, device)

    mAPs = AverageMeter()
    for i, attr in enumerate(cfg.DATA.ATTRIBUTES.NAME):
        mAP = mean_average_precision(q_feats[i], c_feats[i], q_values[i], c_values[i])
        print(f"{attr} mAP: {100. * mAP:.2f}")
        mAPs.update(mAP, q_feats[i].shape[0])

    epoch_avg = mAPs.avg

    print(f"Total MeanAP: {100. * mAPs.avg:.2f}")































