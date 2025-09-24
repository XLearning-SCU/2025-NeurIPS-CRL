import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
from train_mlp import mean_average_precision, AttributeProjectionModel, extract_features, My_DataSet
from torch.utils.data import DataLoader
import argparse
import torch
import numpy as np
from modules.config import cfg
from modules.data.datasets import image_collate_fn
from modules.data.samplers import ImageSampler
from modules.utils.metric import AverageMeter



def parse_args():
    parser = argparse.ArgumentParser(description="Attribute Specific Embedding Network")
    parser.add_argument(
        "--cfg", nargs="+", help="config file", default=['/xlearning/honglin/project/asen++/config/DeepFashion/DeepFashion.yaml'], type=str
    )
    parser.add_argument(
        "--test", help="run test on validation or test set", default=None, type=str
    )
    parser.add_argument(
        "--resume", help="checkpoint model to resume", default='/xlearning/honglin/project/asen++/pretrained_asen', type=str
    )
    parser.add_argument(
        "--seed", default=0, type=int, help="random seed"
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()


    if args.cfg is not None:
        for cfg_file in args.cfg:
            cfg.merge_from_file(cfg_file)
    cfg.freeze()

    device = torch.device(cfg.DEVICE)

    save_path = os.path.join("./data", cfg.DATA.DATASET, "")

    test_candidate_features = np.load(os.path.join(save_path + "test_candidate_features.npy"))
    test_candidate_features = torch.from_numpy(test_candidate_features).to(device).to(torch.float32)
    print(f"test_candidate_features.shape: {test_candidate_features.shape}")

    test_query_features = np.load(os.path.join(save_path + "test_query_features.npy"))
    test_query_features = torch.from_numpy(test_query_features).to(device).to(torch.float32)
    print(f"test_query_features.shape: {test_query_features.shape}")

    text_embedding = np.load(os.path.join(save_path, "text_embedding.npy"))
    text_embedding = torch.from_numpy(text_embedding).to(device).to(torch.float32)
    print(f"text_embedding.shape: {text_embedding.shape}")
    text_embedding = text_embedding.transpose(1, 2)

    gpt_embedding = np.load(os.path.join(save_path, "gpt_embedding.npy"))[:, :500, :]
    gpt_embedding = torch.from_numpy(gpt_embedding).to(device).to(torch.float32)
    print(f"gpt_embedding.shape: {gpt_embedding.shape}")
    gpt_embedding = gpt_embedding.transpose(1, 2)

    common_embedding = np.load(os.path.join(save_path, "common_embedding.npy"))
    common_embedding = torch.from_numpy(common_embedding).to(device).to(torch.float32)
    print(f"common_embedding.shape: {common_embedding.shape}")
    common_embedding = common_embedding.transpose(1, 2)

    text_here = gpt_embedding


    test_set = My_DataSet(cfg, 'TEST')
    test_query_loader = DataLoader(
        test_set,
        collate_fn=image_collate_fn,
        batch_sampler=ImageSampler(cfg, cfg.DATA.GROUNDTRUTH.QUERY['TEST']),
        num_workers=cfg.DATA.NUM_WORKERS,
        shuffle=False,
        pin_memory=True
    )

    test_candidate_loader = DataLoader(
        test_set,
        collate_fn=image_collate_fn,
        batch_sampler=ImageSampler(cfg, cfg.DATA.GROUNDTRUTH.CANDIDATE['TEST']),
        num_workers=cfg.DATA.NUM_WORKERS,
        shuffle=False,
        pin_memory=True
    )

    multi_mlp = AttributeProjectionModel(in_dim=text_here.shape[2], out_dim=1024, n_attr=cfg.DATA.NUM_ATTRIBUTES).to(device)
    checkpoint = torch.load(os.path.join(save_path, "2025-01-21-15:39_best_model.pth"), map_location=device)

    multi_mlp.load_state_dict(checkpoint['model'])
    valid_best_map, valid_best_epoch = checkpoint['mAP'], checkpoint['epoch']
    print(f"Validation best map: {100*valid_best_map: .4f}, epoch: {valid_best_epoch}")

    multi_mlp.eval()

    print("Forwarding query images...")
    q_feats, q_values = extract_features(multi_mlp, test_query_loader, test_query_features, text_here, cfg.DATA.NUM_ATTRIBUTES, n_before=test_candidate_features.shape[0])
    print("Forwarding candidate images...")
    c_feats, c_values = extract_features(multi_mlp, test_candidate_loader, test_candidate_features, text_here, cfg.DATA.NUM_ATTRIBUTES)

    mAPs = AverageMeter()
    for i, attr in enumerate(cfg.DATA.ATTRIBUTES.NAME):
        mAP = mean_average_precision(q_feats[i], c_feats[i], q_values[i], c_values[i])
        print(f"{attr} MAP: {100. * mAP:.2f}")
        mAPs.update(mAP, q_feats[i].shape[0])
    map_avg = mAPs.avg
    print(f"overall MAP: {100*map_avg:.2f}")










