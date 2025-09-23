import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
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
    parser.add_argument("--text_num", default=None, type=int)
    parser.add_argument("--mlp_checkpoint", default="", type=str)
    args = parser.parse_args()
    return args



if __name__ == "__main__":

    args = get_parse()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.cfg is not None:
        for cfg_file in args.cfg:
            cfg.merge_from_file(cfg_file)
    cfg.freeze()
    print(cfg)


    # 获取当前时间
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d-%H:%M")

    device = cfg.DEVICE

    text_path = os.path.join("./data", cfg.DATA.DATASET, "")
    text_embedding = np.load(os.path.join(text_path, "text_embedding.npy"))
    text_embedding = torch.from_numpy(text_embedding).to(device).to(torch.float32)
    print(f"text_embedding.shape: {text_embedding.shape}")
    text_embedding = text_embedding.transpose(1, 2)

    if args.text_num is not None:
        gpt_embedding = np.load(os.path.join(text_path, "gpt_embedding.npy"))[:, :args.text_num, :]
        gpt_embedding = torch.from_numpy(gpt_embedding).to(device).to(torch.float32)
        print(f"gpt_embedding.shape: {gpt_embedding.shape}")
        gpt_embedding = gpt_embedding.transpose(1, 2)
        text_here = gpt_embedding
        save_name = str(args.text_num) + "_only_backbone.pth"
    else:
        common_embedding = np.load(os.path.join(text_path, "common_embedding.npy"))
        common_embedding = torch.from_numpy(common_embedding).to(device).to(torch.float32)
        print(f"common_embedding.shape: {common_embedding.shape}")
        common_embedding = common_embedding.transpose(1, 2)
        text_here = common_embedding
        save_name = "common_text_only_backbone.pth"


    clip_model, preprocess = clip.load('ViT-B/16', device)
    clip_model = clip_model.to(torch.float32)
    for param in clip_model.visual.parameters():
        param.requires_grad = True

    mlp_model = AttributeProjectionModel(in_dim=text_here.shape[2], out_dim=1024, n_attr=cfg.DATA.NUM_ATTRIBUTES).to(device)
    checkpoint = torch.load(os.path.join("/xlearning/honglin/project/asen++/data/DeepFashion", args.mlp_checkpoint), map_location=device, weights_only=False)
    mlp_model.load_state_dict(checkpoint['model'])
    mlp_model = mlp_model.to(torch.float32)
    for param in mlp_model.parameters():
        param.requires_grad = False

    train_loader, valid_query_loader, valid_candidate_loader = build_data(cfg)

    optimizer_clip = torch.optim.Adam(clip_model.parameters(), lr=cfg.SOLVER.CLIP_LR)

    loss_fc = TripletRankingLoss(cfg)

    scaler = torch.amp.GradScaler('cuda')
    scheduler_clip = StepLR(
        optimizer_clip,
        step_size=cfg.SOLVER.STEP_SIZE,
        gamma=cfg.SOLVER.DECAY_RATE
    )

    best_mAP = 0

    mlp_model.eval()
    print(f"only backbone 开始时间：{formatted_time}")
    for epoch in range(cfg.SOLVER.EPOCHS):
        print(f"Epoch {epoch}, lr_clip: {optimizer_clip.param_groups[0]['lr']}")
        clip_model.visual.train()
        # mlp_model.train()

        loss_epoch = 0

        for idx, (x, p, n, a) in tqdm(enumerate(train_loader)):
            a = a.to(device)
            x = torch.stack([preprocess(i) for i in x], dim=0).to(device)
            p = torch.stack([preprocess(i) for i in p], dim=0).to(device)
            n = torch.stack([preprocess(i) for i in n], dim=0).to(device)

            x_img, p_img, n_img = clip_model.encode_image(x), clip_model.encode_image(p), clip_model.encode_image(n)

            x_sim = torch.bmm(x_img.unsqueeze(1), text_here[a]).squeeze(1)
            p_sim = torch.bmm(p_img.unsqueeze(1), text_here[a]).squeeze(1)
            n_sim = torch.bmm(n_img.unsqueeze(1), text_here[a]).squeeze(1)
            x_sim = x_sim / x_sim.norm(dim=-1, keepdim=True)
            p_sim = p_sim / p_sim.norm(dim=-1, keepdim=True)
            n_sim = n_sim / n_sim.norm(dim=-1, keepdim=True)

            new_x, new_p, new_n = mlp_model(x_sim, a), mlp_model(p_sim, a), mlp_model(n_sim, a)
            new_x = new_x / new_x.norm(dim=-1, keepdim=True)
            new_p = new_p / new_p.norm(dim=-1, keepdim=True)
            new_n = new_n / new_n.norm(dim=-1, keepdim=True)

            loss = loss_fc(new_x, new_p, new_n)
            # print(f"loss: {loss.item()}")

            optimizer_clip.zero_grad()
            scaler.scale(loss).backward()

            scaler.step(optimizer_clip)
            scaler.update()


            loss_epoch += loss.item()
        print(f"loss: {loss_epoch:.4f}")

        if (epoch+1) % cfg.SOLVER.EVAL_STEPS == 0:
            clip_model.visual.eval()

            print("Forwarding query images...")
            q_feats, q_values = extract_features(clip_model, mlp_model, valid_query_loader, text_here, cfg.DATA.NUM_ATTRIBUTES, preprocess, device)
            print("Forwarding candidate images...")
            c_feats, c_values = extract_features(clip_model, mlp_model, valid_candidate_loader, text_here, cfg.DATA.NUM_ATTRIBUTES, preprocess, device)

            mAPs = AverageMeter()
            for i, attr in enumerate(cfg.DATA.ATTRIBUTES.NAME):
                mAP = mean_average_precision(q_feats[i], c_feats[i], q_values[i], c_values[i])
                print(f"{attr} mAP: {100. * mAP:.4f}")
                mAPs.update(mAP, q_feats[i].shape[0])

            epoch_avg = mAPs.avg

            print(f"Total MeanAP: {100. * mAPs.avg:.4f}")

            is_best = epoch_avg > best_mAP
            best_mAP = max(epoch_avg, best_mAP)
            if is_best:
                save_path = os.path.join(text_path, save_name)
                state = {
                    'epoch': epoch + 1,
                    'clip_visual': clip_model.visual.state_dict(),
                    'mAP': best_mAP
                }
                torch.save(state, save_path)
        scheduler_clip.step()
    print(f"only backbone 开始时间：{formatted_time}")





























