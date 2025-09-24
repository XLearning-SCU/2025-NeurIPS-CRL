import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
import numpy as np
import torchvision
from tqdm import tqdm
import clip
import sys
import pandas as pd
import argparse
import load_text
import pickle
from sklearn.decomposition import PCA


def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="DeepFashion", help='name of dataset: FashionAI, DeepFashion')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_parse()
    print(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/16', device)

    true_label, prompt, gpt_label = load_text.load_text(args)
    print(f"prompt: {prompt}")
    # print(f"true_labels: {true_label}")
    # print(f"gpt_labels: {gpt_label}")

    # dim_min = min(len(true_label[i]) for i in range(len(prompt)))
    dim_min = min(len(gpt_label[i]) for i in range(len(prompt)))

    text_embedding = []
    gpt_embedding = []
    for k in range(len(prompt)):
        true_label_here = true_label[k]
        gpt_label_here = gpt_label[k]
        prompt_here = prompt[k]

        # text_features = []
        # with torch.no_grad():
        #     for i in true_label_here:
        #         word_with_prompt = prompt_here + i
        #         batch_inputs = clip.tokenize(word_with_prompt).to(device)
        #         batch_features = model.encode_text(batch_inputs)
        #         text_features.extend(batch_features.cpu().numpy())
        # text_features = np.array(text_features)
        # text_features /= np.linalg.norm(text_features, axis=1, keepdims=True)
        # print(f"text_features.shape: {text_features.shape}")

        # if text_features.shape[0] > dim_min:
        #     text = text_features.T
        #     pca = PCA(n_components=dim_min)
        #     reduced_text = pca.fit_transform(text)
        #     text_features = reduced_text.T
        # text_embedding.extend([text_features])

        gpt_features = []
        with torch.no_grad():
            for i in gpt_label_here:
                word_with_prompt = prompt_here + i
                batch_inputs = clip.tokenize(word_with_prompt).to(device)
                batch_features = model.encode_text(batch_inputs)
                gpt_features.extend(batch_features.cpu().numpy())
        gpt_features = np.array(gpt_features)
        gpt_features /= np.linalg.norm(gpt_features, axis=1, keepdims=True)
        print(f"gpt_features.shape: {gpt_features.shape}")
        # gpt_embedding.extend([gpt_features])

        if gpt_features.shape[0] > dim_min:
            text = gpt_features.T
            pca = PCA(n_components=dim_min)
            reduced_text = pca.fit_transform(text)
            gpt_features = reduced_text.T
        gpt_embedding.extend([gpt_features])

    # text_embedding = np.array(text_embedding)
    # print(f"text_embedding.shape: {text_embedding.shape}")
    # save_path = os.path.join("./data", args.dataset, "")
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # np.save(os.path.join(save_path, "text_embedding.npy"), text_embedding)

    gpt_embedding = np.array(gpt_embedding)
    print(f"gpt_embedding.shape: {gpt_embedding.shape}")
    save_path = os.path.join("./data", args.dataset, "")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # np.save(os.path.join(save_path, "gpt_embedding.npy"), gpt_embedding)
    np.save(os.path.join(save_path, "common_embedding.npy"), gpt_embedding)












