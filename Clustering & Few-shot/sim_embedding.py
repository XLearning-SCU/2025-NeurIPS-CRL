import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
from utils.parse import get_parse
from utils import evaluation
import random
from sklearn.cluster import KMeans




def true_testing(embedding, true_text, true_label):
    print("zero-shot: using true_label")
    true_sim = (100 * embedding @ true_text.t()).softmax(dim=-1)
    true_prediction = torch.argmax(true_sim, dim=-1).numpy()
    nmi, ari, _, acc = evaluation.evaluate(true_label, true_prediction)
    print("nmi: {:.2f}, acc: {:.2f}, ari: {:.2f}".format(nmi * 100, acc * 100, ari * 100))


def kmeans_testing(embedding, k, true_label, process=True):
    if isinstance(embedding, np.ndarray):
        new_features = embedding
    elif isinstance(embedding, torch.Tensor):
        new_features = embedding.numpy()
    else:
        raise TypeError("Input data must be a NumPy array or a PyTorch tensor.")

    nmi_list, acc_list, ari_list = [], [], []
    for i in range(20):
        kmeans = KMeans(n_clusters=k, random_state=i, n_init='auto').fit(new_features)
        nmi, ari, _, acc = evaluation.evaluate(true_label, kmeans.labels_)
        if process:
            print("nmi: {:.2f}, acc: {:.2f}, ari: {:.2f}".format(nmi * 100, acc * 100, ari * 100))
        nmi_list.append(nmi * 100)
        acc_list.append(acc * 100)
        ari_list.append(ari * 100)
    nmi_list = np.array(nmi_list)
    acc_list = np.array(acc_list)
    ari_list = np.array(ari_list)
    nmi_mean, acc_mean, ari_mean = np.mean(nmi_list), np.mean(acc_list), np.mean(ari_list)
    print(f"nmi_mean: {nmi_mean: .2f}, acc_mean: {acc_mean: .2f}, ari_mean: {ari_mean: .2f}")
    return nmi_mean, acc_mean, ari_mean


if __name__ == "__main__":
    args = get_parse()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    save_path = os.path.join("./data", args.dataset, '')
    img_features = np.load(save_path + args.criterion + "_image_embedding.npy")
    # print(f"img_features.shape: {img_features.shape}")
    common_texts = np.load(save_path + args.criterion + "_common_texts.npy")
    # print(f"common_texts.shape: {common_texts.shape}")
    # gpt_texts = np.load(save_path + args.criterion + "_gpt_texts.npy")[:args.all_text]
    # print(f"gpt_texts.shape: {gpt_texts.shape}")
    img_labels = np.loadtxt(save_path + args.criterion + "_labels.txt")
    # print(f"img_labels.shape: {img_labels.shape}")

    cluster_num = len(set(img_labels))
    # print(f"cluster_num: {cluster_num}")
    sample_num = len(img_labels)
    # print(f"sample_num: {sample_num}")

    common_texts = torch.from_numpy(common_texts).type(torch.float32)
    # gpt_texts = torch.from_numpy(gpt_texts).type(torch.float32)
    img_features = torch.from_numpy(img_features).type(torch.float32)
    print("original image features:")
    kmeans_testing(img_features, cluster_num, img_labels)

    # sim_features = img_features @ gpt_texts.T
    sim_features = img_features @ common_texts.T
    print(sim_features.shape)

    sim_features /= torch.norm(sim_features, p=2, dim=1, keepdim=True)
    print("new features:")
    kmeans_testing(sim_features, cluster_num, img_labels, process=True)

































