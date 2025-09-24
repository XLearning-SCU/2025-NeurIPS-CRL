# 2025-NeurIPS-CRL

# Similarity Retrieval

## Dataset

We adopt [GeneCIS](https://github.com/facebookresearch/genecis) as the benchmark dataset, which can be downloaded according to [instructions](https://github.com/facebookresearch/genecis/blob/main/DOWNLOAD.md). 

We only use the "Focus on an Object" and "Change an Object" settings, thus you only need to download the small COCO subset. 

Pre-trained CLIP model weights can also be downloaded from that page, and we only use the ViT-B/16 version.

## Evaluation

Change the parameter '--pretrained' in parse.py to get the original/pre-trained CLIP model.

Change the parameter '--dataset' in parse.py to get the "change_object"/"focus_object" setting.

Generate the text embeddings:

> python text_embedding.py

Evaluate the method by:

> cd eval/
> 
> python evaluate.py
