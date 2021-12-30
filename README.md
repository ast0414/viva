# ViVA: Semi-supervised Visualization via Variational Autoencoders
This repository contains the source code for the paper:
***An, Sungtae, et al. "ViVA: Semi-supervised Visualization via Variational Autoencoders." 2020 IEEE International Conference on Data Mining (ICDM). IEEE, 2020.***

### Tested with Pytorch 1.9.1 (cudatoolkit 11.1) on Python 3.7

0. Create a (conda) virtual environment and install the requirements
```
conda env create -f environment.yml
```

1. Create a partially labeled dataset (MNIST)
```
python mnist_create_semisupervised_dataset.py --n-labeled 1000
```

2. Train ViVA
```
python mnist_train_viva.py --dataset Data/mnist_1000_labeled.pkl
```

3. Visualize the embeddings
```
python mnist_visualize_embeddings.py --model Results/best_checkpoint.pth
```
