import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn as nn
import argparse
from modules.config import cfg
from modules.utils.logger import setup_logger
import clip
import numpy as np
from torch.utils.data import DataLoader
from modules.data.datasets import BaseDataSet, image_collate_fn
from modules.data.samplers import ImageSampler
from tqdm import tqdm
from image_loader import ImageLoader




def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description="Attribute Specific Embedding Network")
    parser.add_argument(
        "--cfg", nargs="+", help="config file", default=['/xlearning/honglin/project/asen++/config/DeepFashion/DeepFashion.yaml'], type=str
    )
    parser.add_argument(
        "--test", help="run test on validation or test set", default=None, type=str
    )
    return parser.parse_args()


@torch.no_grad()
def extract_features(clip_model, data_loader, preprocess):
    feats = []
    with tqdm(total=len(data_loader)) as bar:
        for idx, batch in enumerate(data_loader):
            x, a, v = batch
            img = torch.stack([preprocess(i) for i in x], dim=0).cuda()

            out = clip_model.encode_image(img)
            feats.extend(out.cpu().numpy())

            bar.update(1)
    feats = np.array(feats)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    return feats


if __name__ == "__main__":
    args = parse_args()
    print(args)
    if args.cfg is not None:
        for cfg_file in args.cfg:
            cfg.merge_from_file(cfg_file)
    cfg.freeze()

    logger = setup_logger(name="ASEN", level=cfg.LOGGER.LEVEL, stream=cfg.LOGGER.STREAM)
    logger.info(cfg)

    # clip_model, preprocess = clip.load("RN50", device='cuda')
    clip_model, preprocess = clip.load("ViT-B/16", device='cuda')
    save_path = os.path.join("./data", cfg.DATA.DATASET, "")
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    train_loader = torch.utils.data.DataLoader(
            ImageLoader(cfg.DATA.BASE_PATH, cfg.DATA.DATASET, 'filenames_train.txt',
                        'train', 'label',
                        transform=preprocess
                        ),
            batch_size=100, shuffle=False, drop_last=False, num_workers=4
        )

    train_features = []
    with torch.no_grad():
        for img, _, _, _ in tqdm(train_loader):
            img = img.cuda()
            batch_features = clip_model.encode_image(img)
            train_features.extend(batch_features.cpu().numpy())
    train_features = np.array(train_features)
    train_features /= np.linalg.norm(train_features, axis=1, keepdims=True)
    print(f"train_features.shape: {train_features.shape}")
    np.save(os.path.join(save_path, "train_features.npy"), train_features)

    test_candidate_loader = torch.utils.data.DataLoader(
        ImageLoader(cfg.DATA.BASE_PATH, cfg.DATA.DATASET, 'filenames_test.txt',
                    'test', 'candidate',
                    transform=preprocess
                    ),
        batch_size=100, shuffle=False, drop_last=False, num_workers=4)

    test_query_loader = torch.utils.data.DataLoader(
        ImageLoader(cfg.DATA.BASE_PATH, cfg.DATA.DATASET, 'filenames_test.txt',
                    'test', 'query',
                    transform=preprocess
                    ),
        batch_size=100, shuffle=False, drop_last=False, num_workers=4)

    test_candidate_features = []
    test_query_features = []

    with torch.no_grad():
        for img, _, _, _ in tqdm(test_candidate_loader):
            img = img.cuda()
            batch_features = clip_model.encode_image(img)
            test_candidate_features.extend(batch_features.cpu().numpy())
    test_candidate_features = np.array(test_candidate_features)
    test_candidate_features /= np.linalg.norm(test_candidate_features, axis=1, keepdims=True)
    print(f"test_candidate_features.shape: {test_candidate_features.shape}")
    np.save(save_path + "test_candidate_features.npy", test_candidate_features)

    with torch.no_grad():
        for img, _, _, _ in tqdm(test_query_loader):
            img = img.cuda()
            batch_features = clip_model.encode_image(img)
            test_query_features.extend(batch_features.cpu().numpy())
    test_query_features = np.array(test_query_features)
    test_query_features /= np.linalg.norm(test_query_features, axis=1, keepdims=True)
    print(f"test_query_features.shape: {test_query_features.shape}")
    np.save(save_path + "test_query_features.npy", test_query_features)


    valid_candidate_loader = torch.utils.data.DataLoader(
            ImageLoader(cfg.DATA.BASE_PATH, cfg.DATA.DATASET, 'filenames_valid.txt',
                        'valid', 'candidate',
                        transform=preprocess
                        ),
            batch_size=100, shuffle=False, drop_last=False, num_workers=4)

    valid_query_loader = torch.utils.data.DataLoader(
        ImageLoader(cfg.DATA.BASE_PATH, cfg.DATA.DATASET, 'filenames_valid.txt',
                    'valid', 'query',
                    transform=preprocess
                    ),
        batch_size=100, shuffle=False, drop_last=False, num_workers=4)

    valid_candidate_features = []
    valid_query_features = []

    with torch.no_grad():
        for img, _, _, _ in tqdm(valid_candidate_loader):
            img = img.cuda()
            batch_features = clip_model.encode_image(img)
            valid_candidate_features.extend(batch_features.cpu().numpy())
    valid_candidate_features = np.array(valid_candidate_features)
    valid_candidate_features /= np.linalg.norm(valid_candidate_features, axis=1, keepdims=True)
    print(f"valid_candidate_features.shape: {valid_candidate_features.shape}")
    np.save(save_path + "valid_candidate_features.npy", valid_candidate_features)

    with torch.no_grad():
        for img, _, _, _ in tqdm(valid_query_loader):
            img = img.cuda()
            batch_features = clip_model.encode_image(img)
            valid_query_features.extend(batch_features.cpu().numpy())
    valid_query_features = np.array(valid_query_features)
    valid_query_features /= np.linalg.norm(valid_query_features, axis=1, keepdims=True)
    print(f"valid_query_features.shape: {valid_query_features.shape}")
    np.save(save_path + "valid_query_features.npy", valid_query_features)









