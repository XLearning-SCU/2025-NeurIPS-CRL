import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
from utils.parse import get_parse
import random
from utils.more_data import data_ablation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sim_embedding import replace_with_iterative_removal
import clip


def few_shot_split(img_features, img_labels, seed_i, shots=1):
    np.random.seed(seed_i)
    unique_classes = np.unique(img_labels)
    train_indices = []
    test_indices = []

    for cls in unique_classes:
        cls_indices = np.where(img_labels == cls)[0]
        np.random.shuffle(cls_indices)
        train_indices.extend(cls_indices[:shots])
        test_indices.extend(cls_indices[shots:])

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    train_features = img_features[train_indices]
    train_labels = img_labels[train_indices]
    test_features = img_features[test_indices]
    test_labels = img_labels[test_indices]

    return train_features, train_labels, test_features, test_labels


def linear_probe(train_features, train_labels, test_features, test_labels):
    clf = LogisticRegression(max_iter=1000)
    # clf = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=0)
    clf.fit(train_features, train_labels)
    predictions = clf.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)

    return accuracy



if __name__ == "__main__":
    args = get_parse()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load('ViT-B/32', device)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset_path, true_label, gpt_label, prompt = data_ablation(args)
    print(f"prompt: {prompt}")
    # print(f"true_labels: {true_label}")

    save_path = os.path.join("./data", args.dataset, '')
    img_features = np.load(save_path + args.criterion + "_image_embedding.npy")
    # print(f"img_features.shape: {img_features.shape}")
    common_texts = np.load(save_path + args.criterion + "_common_texts.npy")
    # print(f"common_texts.shape: {common_texts.shape}")
    # gpt_texts = np.load(save_path + args.criterion + "_gpt_texts.npy")[:args.all_text]
    # print(f"gpt_texts.shape: {gpt_texts.shape}")
    img_labels = np.loadtxt(save_path + args.criterion + "_labels.txt")
    # print(f"img_labels.shape: {img_labels.shape}")

    acc_origin_list = []
    for seed_i in range(20):
        train_features, train_labels, test_features, test_labels = few_shot_split(img_features, img_labels, seed_i, args.shots)
        accuracy = linear_probe(train_features, train_labels, test_features, test_labels)
        print(f"Accuracy for seed {seed_i}: {accuracy*100: .2f}")
        acc_origin_list.append(accuracy*100)
    acc_origin = np.mean(acc_origin_list)
    print(f"Origin Accuracy: {acc_origin: .2f}")
    # exit(0)

    # gpt_texts = torch.from_numpy(gpt_texts).type(torch.float32)
    common_texts = torch.from_numpy(common_texts).type(torch.float32)
    img_features = torch.from_numpy(img_features).type(torch.float32)

    # sim_features = img_features @ gpt_texts.T
    sim_features = img_features @ common_texts.T
    sim_features = sim_features - torch.mean(sim_features)
    sim_features /= torch.std(sim_features, dim=1, keepdim=True)
    print(f"sim_features.shape: {sim_features.shape}")
    sim_features = sim_features.numpy()

    acc_sim_list = []
    for seed_i in range(20):
        train_features, train_labels, test_features, test_labels = few_shot_split(sim_features, img_labels, seed_i, args.shots)
        accuracy = linear_probe(train_features, train_labels, test_features, test_labels)
        print(f"Accuracy for seed {seed_i}: {accuracy * 100:.2f}")
        acc_sim_list.append(accuracy*100)
    acc_sim = np.mean(acc_sim_list)
    print(f"Sim Accuracy: {acc_sim: .2f}")








