import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from modules.config import cfg
from modules.utils.logger import setup_logger
from modules.solver import build_optimizer, build_lr_scheduler
from modules.loss import build_loss
import clip
from modules.data.datasets import image_collate_fn
from modules.data.samplers import ImageSampler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
from torch.utils.data.sampler import Sampler
import math
from modules.utils.metric import APScorer, AverageMeter


def mean_average_precision(queries, candidates, q_values, c_values):
    scorer = APScorer(candidates.shape[0])

    # similarity matrix
    simmat = np.matmul(queries, candidates.T)

    ap_sum = 0
    for q in range(simmat.shape[0]):
        sim = simmat[q]
        index = np.argsort(sim)[::-1]
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


class AttributeProjectionModel(nn.Module):
    def __init__(self, in_dim, out_dim, n_attr):
        super().__init__()
        self.n_attr = n_attr
        self.projection_heads = nn.ModuleList([MLP(in_dim, out_dim) for _ in range(n_attr)])
        # self.projection_heads = nn.ModuleList([MLP_3(in_dim, out_dim) for _ in range(n_attr)])

    def forward(self, x, attr_idx):
        assert torch.max(attr_idx) < self.n_attr, "Attribute index out of range"
        # Initialize output tensor
        outputs = torch.zeros((x.size(0), self.projection_heads[0].fc[-1].out_features), device=x.device)
        # Process each sample based on its attribute index
        for i in range(self.n_attr):
            mask = (attr_idx == i)
            if mask.any() and int(mask.sum())>1:
                outputs[mask] = self.projection_heads[i](x[mask])
        return outputs


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.hidden_dim = int((in_dim + out_dim) / 2)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, out_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class MLP_3(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.hidden_dim = int((in_dim + out_dim) / 2)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, out_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        return x




class TripletSampler(Sampler):
    def __init__(self, cfg):
        self.num_triplets = cfg.DATA.NUM_TRIPLETS
        self.batch_size = cfg.DATA.TRAIN_BATCHSIZE
        self.attrs = cfg.DATA.ATTRIBUTES.NAME
        self.num_values = cfg.DATA.ATTRIBUTES.NUM
        self.indices = {}
        for i, attr in enumerate(self.attrs):
            self.indices[attr] = [[] for _ in range(self.num_values[i])]

        label_file = os.path.join(cfg.DATA.BASE_PATH, cfg.DATA.DATASET, cfg.DATA.GROUNDTRUTH.TRAIN)
        assert os.path.exists(label_file), f"Train label file {label_file} does not exist."
        with open(label_file, 'r') as f:
            for l in f:
                l = [int(i) for i in l.strip().split()]
                fid = l[0]
                attr_val = [(l[i], l[i+1]) for i in range(1, len(l), 2)]
                for attr, val in attr_val:
                    self.indices[self.attrs[attr]][val].append(fid)


    def __len__(self):
        return math.ceil(self.num_triplets / self.batch_size)

    def __str__(self):
        return f"| Triplet Sampler | iters {self.__len__()} | batch size {self.batch_size}|"

    def __iter__(self):
        """
            每次迭代生成一个批次的数据。
            """
        sampled_attrs = random.choices(range(0, len(self.attrs)), k=self.num_triplets)
        for i in range(self.__len__()):
            attrs = sampled_attrs[i * self.batch_size:(i + 1) * self.batch_size]

            anchors = []
            positives = []
            negatives = []
            for a in attrs:
                # 随机选择两个属性值
                vp, vn = random.sample(range(self.num_values[a]), 2)
                # 随机选择 anchor 和 positive 样本
                x, p = random.sample(self.indices[self.attrs[a]][vp], 2)
                # 随机选择 negative 样本
                n = random.choice(self.indices[self.attrs[a]][vn])
                # 将数据格式化为 (图像索引, 属性索引) 的形式
                anchors.append((x, a))
                positives.append((p, a))
                negatives.append((n, a))

            yield anchors + positives + negatives


def triplet_collate_fn(batch):
    """
    聚合 DataLoader 返回的数据，使其直接以张量形式返回。
    """
    n = len(batch) // 3
    x, x_a = zip(*batch[:n])
    p, p_a = zip(*batch[n:2*n])
    n, n_a = zip(*batch[2*n:3*n])

    # 直接将图像数据堆叠为张量
    # x = torch.stack(x)
    # p = torch.stack(p)
    # n = torch.stack(n)

    return x, p, n, torch.LongTensor(x_a)


class My_DataSet(Dataset):
    def __init__(self, cfg, split):
        self.root_path = os.path.join(cfg.DATA.BASE_PATH, cfg.DATA.DATASET)

        self.fnamelist = []
        filepath = os.path.join(self.root_path, cfg.DATA.PATH_FILE[split])
        assert os.path.exists(filepath), f"File {filepath} does not exist."
        with open(filepath, 'r') as f:
            for l in f:
                self.fnamelist.append(l.strip())

    def __len__(self):
        return self.fnamelist

    def __getitem__(self, index):
        img_index = index[0]

        return (img_index,) + index[1:]


@torch.no_grad()
def extract_features(model, data_loader, features, text, n_attrs, n_before=0):
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

            x_sim = x_sim / x_sim.norm(dim=-1, keepdim=True)

            # x_sim = x_sim - torch.mean(x_sim)
            # x_sim = x_sim / torch.std(x_sim, dim=-1, keepdim=True)

            out = model(x_sim, a)

            feats.append(out.detach().cpu().numpy())
            values.append(v.numpy())

            for i in range(a.size(0)):
                indices[a[i].cpu().item()].append(cnt)
                cnt += 1

            bar.update(1)

    feats = np.concatenate(feats)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    values = np.concatenate(values)

    feats = [feats[indices[i]] for i in range(n_attrs)]
    values = [values[indices[i]] for i in range(n_attrs)]

    return feats, values




def main(cfg):
    # 获取当前时间
    current_time = datetime.now()
    # 格式化时间
    formatted_time = current_time.strftime("%Y-%m-%d-%H:%M")
    save_name = formatted_time + "_best_model.pth"

    logger = setup_logger(name="ASEN", level=cfg.LOGGER.LEVEL, stream=cfg.LOGGER.STREAM)
    logger.info(cfg)

    device = torch.device(cfg.DEVICE)

    save_path = os.path.join("./data", cfg.DATA.DATASET, "")

    train_features = np.load(os.path.join(save_path, "train_features.npy"))
    train_features = torch.from_numpy(train_features).to(device).to(torch.float32)
    print(f"train_features.shape: {train_features.shape}")

    valid_candidate_features = np.load(os.path.join(save_path + "valid_candidate_features.npy"))
    valid_candidate_features = torch.from_numpy(valid_candidate_features).to(device).to(torch.float32)
    print(f"valid_candidate_features.shape: {valid_candidate_features.shape}")

    valid_query_features = np.load(os.path.join(save_path + "valid_query_features.npy"))
    valid_query_features = torch.from_numpy(valid_query_features).to(device).to(torch.float32)
    print(f"valid_query_features.shape: {valid_query_features.shape}")

    # text_embedding = np.load(os.path.join(save_path, "text_embedding.npy"))
    # text_embedding = torch.from_numpy(text_embedding).to(device).to(torch.float32)
    # print(f"text_embedding.shape: {text_embedding.shape}")
    # text_embedding = text_embedding.transpose(1, 2)

    # gpt_embedding = np.load(os.path.join(save_path, "gpt_embedding.npy"))[:, :500, :]
    # gpt_embedding = torch.from_numpy(gpt_embedding).to(device).to(torch.float32)
    # print(f"gpt_embedding.shape: {gpt_embedding.shape}")
    # gpt_embedding = gpt_embedding.transpose(1, 2)

    gpt_embedding = np.load(os.path.join(save_path, "common_embedding.npy"))
    gpt_embedding = torch.from_numpy(gpt_embedding).to(device).to(torch.float32)
    print(f"common_embedding.shape: {gpt_embedding.shape}")
    gpt_embedding = gpt_embedding.transpose(1, 2)

    multi_mlp = AttributeProjectionModel(in_dim=gpt_embedding.shape[2], out_dim=1024, n_attr=cfg.DATA.NUM_ATTRIBUTES).to(device)
    # mlp_model = MLP(in_dim=text_embedding.shape[2], out_dim=1024).to(device)

    train_set = My_DataSet(cfg, 'TRAIN')
    train_loader = DataLoader(
        train_set,
        collate_fn=triplet_collate_fn,
        batch_sampler=TripletSampler(cfg),
        num_workers=cfg.DATA.NUM_WORKERS,
        pin_memory=True
    )

    valid_set = My_DataSet(cfg, 'VALID')
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

    optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(
        multi_mlp.parameters(),
        lr=cfg.SOLVER.BASE_LR
    )
    scheduler = build_lr_scheduler(cfg, optimizer)
    criterion = build_loss(cfg)

    best_mAP = 0
    for epoch in range(cfg.SOLVER.EPOCHS):
        multi_mlp.train()
        print(f"epoch: {epoch}, lr: {optimizer.param_groups[0]['lr']}")

        loss_epoch = 0
        for idx, (x, p, n, a) in tqdm(enumerate(train_loader)):
            x_img = train_features[x]
            p_img = train_features[p]
            n_img = train_features[n]

            x_sim = torch.bmm(x_img.unsqueeze(1), gpt_embedding[a]).squeeze(1)
            p_sim = torch.bmm(p_img.unsqueeze(1), gpt_embedding[a]).squeeze(1)
            n_sim = torch.bmm(n_img.unsqueeze(1), gpt_embedding[a]).squeeze(1)

            # x_sim = x_sim - torch.mean(x_sim)
            # x_sim = x_sim / torch.std(x_sim, dim=-1, keepdim=True)
            # p_sim = p_sim - torch.mean(p_sim)
            # p_sim = p_sim / torch.std(p_sim, dim=-1, keepdim=True)
            # n_sim = n_sim - torch.mean(n_sim)
            # n_sim = n_sim / torch.std(n_sim, dim=-1, keepdim=True)

            x_sim = x_sim / x_sim.norm(dim=-1, keepdim=True)
            p_sim = p_sim / p_sim.norm(dim=-1, keepdim=True)
            n_sim = n_sim / n_sim.norm(dim=-1, keepdim=True)

            new_x, new_p, new_n = multi_mlp(x_sim, a), multi_mlp(p_sim, a), multi_mlp(n_sim, a)
            new_x = new_x / new_x.norm(dim=-1, keepdim=True)
            new_p = new_p / new_p.norm(dim=-1, keepdim=True)
            new_n = new_n / new_n.norm(dim=-1, keepdim=True)

            loss = criterion(new_x, new_p, new_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()
        print(f"Loss: {loss_epoch: .4f}")


        if (epoch+1) % cfg.SOLVER.EVAL_STEPS == 0:
            multi_mlp.eval()
            logger.info("Forwarding query images...")
            q_feats, q_values = extract_features(multi_mlp, valid_query_loader, valid_query_features, gpt_embedding, len(cfg.DATA.ATTRIBUTES.NAME), n_before=valid_candidate_features.shape[0])
            logger.info("Forwarding candidate images...")
            c_feats, c_values = extract_features(multi_mlp, valid_candidate_loader, valid_candidate_features, gpt_embedding, len(cfg.DATA.ATTRIBUTES.NAME))

            mAPs = AverageMeter()
            for i, attr in enumerate(cfg.DATA.ATTRIBUTES.NAME):
                mAP = mean_average_precision(q_feats[i], c_feats[i], q_values[i], c_values[i])
                logger.info(f"{attr} mAP: {100. * mAP:.4f}")
                mAPs.update(mAP, q_feats[i].shape[0])

            epoch_avg = mAPs.avg

            logger.info(f"Total MeanAP: {100. * mAPs.avg:.4f}")

            is_best = epoch_avg > best_mAP
            best_mAP = max(epoch_avg, best_mAP)
            if is_best:
                state = {
                    'epoch': epoch+1,
                    'model': multi_mlp.state_dict(),
                    'mAP': best_mAP
                }
                torch.save(state, os.path.join(save_path, save_name))

        scheduler.step()

    print(f"开始时间：{formatted_time}")














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
    parser.add_argument(
        "--resume", help="checkpoint model to resume", default='/xlearning/honglin/project/asen++/pretrained_asen', type=str
    )
    parser.add_argument(
        "--seed", default=4, type=int, help="random seed"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.cfg is not None:
        for cfg_file in args.cfg:
            cfg.merge_from_file(cfg_file)
    cfg.freeze()
    main(cfg)








