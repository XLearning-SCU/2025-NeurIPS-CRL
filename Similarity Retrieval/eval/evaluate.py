import os
from datasets.vaw_dataset import VAWValSubset, VAWValFocus, VAWValChange
from datasets.coco_dataset import COCOValSubset
from eval.eval_functions import validate
import clip
import torch
import argparse
import torch.backends.cudnn as cudnn
from functools import partial
import config as cfg
from parse import get_parse
from gen_utils import strip_state_dict
import numpy as np
from combiner_model import Combiner, FeatureComb





def main(args):

    print(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/16", device=device)
    clip_model.float().eval()
    input_dim = clip_model.visual.input_resolution
    feature_dim = clip_model.visual.output_dim

    print('Loading datasets...')
    tokenizer = partial(clip.tokenize, truncate=True)
    genecis_split_path = os.path.join(cfg.genecis_root, f'{args.dataset}.json')

    print(f'Evaluating on GeneCIS {args.dataset} from {genecis_split_path}')
    val_dataset_subset = COCOValSubset(val_split_path=genecis_split_path, tokenizer=tokenizer, transform=preprocess)
    print(f'Evaluating on {len(val_dataset_subset)} templates...')


    val_loader = torch.utils.data.DataLoader(
        val_dataset_subset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False
    )

    save_path = os.path.join("../data", args.dataset, '')
    # load fine-tuned CLIP backbone
    if args.pretrained:
        print("Loading pretrained weights--------------")
        state_dict = torch.load(cfg.clip_backbone, map_location='cpu', weights_only=True)
        state_dict = strip_state_dict(state_dict=state_dict, strip_key='module.')
        clip_model.load_state_dict(state_dict)
        # gpt_texts = np.load(save_path + "pretrained_texts.npy")[:args.all_text]
        gpt_texts = np.load(save_path + "pretrained_texts.npy")
        gpt_texts = torch.from_numpy(gpt_texts).type(torch.float32).to(device)
        print(f"gpt_text.shape: {gpt_texts.shape}")
    else:
        # gpt_texts = np.load(save_path + "gpt_texts.npy")[:args.all_text]
        gpt_texts = np.load(save_path + "gpt_texts.npy")
        gpt_texts = torch.from_numpy(gpt_texts).type(torch.float32).to(device)
        print(f"gpt_text.shape: {gpt_texts.shape}")


    if args.combiner_mode == 'combiner_original':
        combiner = Combiner(clip_feature_dim=feature_dim, projection_dim=2560, hidden_dim=2 * 2560)
        state_dict = torch.load(cfg.combiner, map_location='cpu')
        state_dict = strip_state_dict(state_dict=state_dict, strip_key='module.')
        combiner.load_state_dict(state_dict)
        combiner = combiner.to(device)
        validate(clip_model, combiner, val_loader)

    elif args.combiner_mode in ('image_only', 'text_only', 'image_plus_text'):
        combiner = FeatureComb(args.combiner_mode)
        validate(clip_model, combiner, val_loader, gpt_texts)

    else:
        raise ValueError


if __name__ == '__main__':

    args = get_parse()
    main(args)























