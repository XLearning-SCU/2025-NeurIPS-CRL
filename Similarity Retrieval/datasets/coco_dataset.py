# Copyright (c) Meta Platforms, Inc. and affiliates.

from torch.utils.data import Dataset
import os
from PIL import Image
import json
import torch
import numpy as np
import config as cfg


class COCODataset(Dataset):

    def __init__(self, transform=None, root_dir=cfg.coco_root) -> None:
        super().__init__()

        self.root_dir = root_dir
        self.transform = transform

    def load_sample(self, sample):

        val_img_id = sample['val_image_id']
        fpath = os.path.join(self.root_dir, f'{val_img_id:012d}.jpg')
        img = Image.open(fpath)
        
        if self.transform is not None:
            img = self.transform(img)

        return img

class COCOValSubset(COCODataset):

    def __init__(self, val_split_path, tokenizer=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        with open(val_split_path) as f:
            val_samples = json.load(f)

        self.val_samples = val_samples
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        
        """
        Follow same return signature as CIRRSubset
        """

        sample = self.val_samples[index]
        reference = sample['reference']

        target = sample['target']
        gallery = sample['gallery']
        caption = sample['condition']

        caption_text = caption

        caption = 'a photo with a ' + caption
        positions = ["left", "right", "top", "lower", "mid"]
        caption = [caption + ' on the ' + position for position in positions]

        reference, target = [self.load_sample(i) for i in (reference, target)]
        gallery = [self.load_sample(i) for i in gallery]

        if self.transform is not None:
            gallery = torch.stack(gallery)
            gallery_and_target = torch.cat([target.unsqueeze(0), gallery])
        else:
            gallery_and_target = [target] + gallery

        if self.tokenizer is not None:
            for num, j in enumerate(caption):
                caption[num] = self.tokenizer(j).squeeze().numpy()
        caption_token = np.array(caption)
        caption_token = torch.tensor(caption_token)

        return reference, gallery_and_target, 0, caption_text, caption_token

    def __len__(self):
        return len(self.val_samples)
