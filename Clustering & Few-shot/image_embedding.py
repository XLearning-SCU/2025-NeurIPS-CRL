import os
import torch
import numpy as np
import torchvision
from tqdm import tqdm
from utils.parse import get_parse
from utils.more_data import data_ablation
from utils import clevr4
import clip


if __name__ == "__main__":
    args = get_parse()
    print(args)

    dataset_path, true_label, gpt_label, prompt = data_ablation(args)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    if args.dataset == "clevr4":
        dataset = clevr4.Clevr4(
            root=dataset_path,
            taxonomy=args.criterion,
            transform=preprocess
        )
    else:
        dataset = torchvision.datasets.ImageFolder(
            root=dataset_path,
            transform=preprocess
        )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=100,
        shuffle=False,
        drop_last=False,
        num_workers=4
    )

    img_features = []
    img_label = []

    with torch.no_grad():
        for imgs, labels in tqdm(data_loader):
            imgs = imgs.to(device)
            batch_features = model.encode_image(imgs)
            img_label.extend(labels.numpy())
            img_features.extend(batch_features.cpu().numpy())
    img_features = np.array(img_features)
    img_label = np.array(img_label)
    img_features /= np.linalg.norm(img_features, axis=1, keepdims=True)
    print(f"img_features.shape: {img_features.shape}")
    print(f"img_label.shape: {img_label.shape}")

    save_path = os.path.join("./data", args.dataset, '')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(save_path + args.criterion + "_image_embedding.npy", img_features)
    np.savetxt(save_path + args.criterion + "_labels.txt", img_label)
