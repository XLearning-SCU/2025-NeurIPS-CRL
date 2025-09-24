import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
from train_mlp import mean_average_precision, My_DataSet
from torch.utils.data import DataLoader
import argparse
import torch
import numpy as np
from modules.config import cfg
from modules.data.datasets import image_collate_fn
from modules.data.samplers import ImageSampler
from modules.utils.metric import AverageMeter
from tqdm import tqdm
from scipy.special import softmax


def extract_features(data_loader, features, text, n_attrs, n_before=0):
    feats = []
    indices = [[] for _ in range(n_attrs)]
    values = []
    with tqdm(total=len(data_loader)) as bar:
        cnt = 0
        for idx, batch in enumerate(data_loader):
            x, a, v = batch
            index = [i - n_before for i in x]
            img_features = features[index]
            x_sim = torch.bmm(img_features.unsqueeze(1), text[a]).squeeze(1)
            feats.append(x_sim.detach().cpu().numpy())
            values.append(v.numpy())
            for i in range(a.size(0)):
                indices[a[i].cpu().item()].append(cnt)
                cnt += 1
            bar.update(1)
    feats = np.concatenate(feats)
    # 过一次激活函数，softmax或者其他
    # feats = softmax(feats, axis=1)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    print(feats.shape)
    values = np.concatenate(values)
    feats = [feats[indices[i]] for i in range(n_attrs)]
    values = [values[indices[i]] for i in range(n_attrs)]
    return feats, values


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

    # text_embedding = np.load(os.path.join(save_path, "text_embedding.npy"))
    # text_embedding = torch.from_numpy(text_embedding).to(device).to(torch.float32)
    # print(f"text_embedding.shape: {text_embedding.shape}")
    # text_embedding = text_embedding.transpose(1, 2)

    # gpt_embedding = np.load(os.path.join(save_path, "gpt_embedding.npy"))[:, :1000, :]
    # gpt_embedding = torch.from_numpy(gpt_embedding).to(device).to(torch.float32)
    # print(f"gpt_embedding.shape: {gpt_embedding.shape}")
    # gpt_embedding = gpt_embedding.transpose(1, 2)

    common_embedding = np.load(os.path.join(save_path, "common_embedding.npy"))
    common_embedding = torch.from_numpy(common_embedding).to(device).to(torch.float32)
    print(f"common_embedding.shape: {common_embedding.shape}")
    common_embedding = common_embedding.transpose(1, 2)

    text_here = common_embedding

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

    print("Forwarding query images...")
    q_feats, q_values = extract_features(test_query_loader, test_query_features, text_here, cfg.DATA.NUM_ATTRIBUTES, n_before=test_candidate_features.shape[0])
    print("Forwarding candidate images...")
    c_feats, c_values = extract_features(test_candidate_loader, test_candidate_features, text_here, cfg.DATA.NUM_ATTRIBUTES)


    mAPs = AverageMeter()
    for i, attr in enumerate(cfg.DATA.ATTRIBUTES.NAME):
        mAP = mean_average_precision(q_feats[i], c_feats[i], q_values[i], c_values[i])
        print(f"{attr} MAP: {100. * mAP:.2f}")
        mAPs.update(mAP, q_feats[i].shape[0])
    map_avg = mAPs.avg
    print(f"overall MAP: {100*map_avg:.2f}")










