# 2025-NeurIPS-CRL

# Clustering & Few-shot

## Configuration

1. Edit the configuration file "utils/parse.py".
2. Edit the dataset path in "more_data.py".

## Dataset

Download datasets from the following URLs: [Clevr4-10k](https://www.robots.ox.ac.uk/~vgg/data/clevr4/), [Cards](http://faculty.washington.edu/juhuah/images/AugDMC_datasets.zip).

Run the following commands to obtain the image and text embeddings:

> python image_embedding.py

> python text_embedding.py

## Clustering

> python sim_embedding.py

## Few-shot Classification

> python linear_probe.py

