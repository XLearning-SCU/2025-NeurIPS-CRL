import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
from tqdm import tqdm
import clip
from utils.more_data import data_ablation
from utils import other_llm, llm_prompt
from utils.parse import get_parse



if __name__ == "__main__":
    args = get_parse()
    print(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    dataset_path, _, gpt_label, prompt = data_ablation(args)
    # dataset_path, _, gpt_label, prompt = other_llm.data_ablation(args)
    # dataset_path, _, gpt_label, prompt = llm_prompt.data_ablation(args)
    print(f"prompt: {prompt}")
    # print(f"gpt_labels: {gpt_label}")
    print(len(gpt_label))

    gpt_texts = []
    only_texts = []

    with torch.no_grad():
        for j in tqdm(gpt_label):
            gpt_word_with_prompt = prompt + j
            gpt_batch_inputs = clip.tokenize(gpt_word_with_prompt).to(device)
            gpt_batch = model.encode_text(gpt_batch_inputs)
            gpt_texts.extend(gpt_batch.cpu().numpy())
    gpt_texts = np.array(gpt_texts)
    gpt_texts /= np.linalg.norm(gpt_texts, axis=1, keepdims=True)
    print(f"gpt_texts.shape: {gpt_texts.shape}")


    save_path = os.path.join("./data", args.dataset, '')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # np.save(save_path + args.criterion + "_gpt_texts.npy", gpt_texts)
    np.save(save_path + args.criterion + "_common_texts.npy", gpt_texts)










