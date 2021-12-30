import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models import MnistViVA


def main(args):
    print('===> Loaded Configuration')
    print(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    cudnn.benchmark = True if args.cuda else False
    device = torch.device("cuda" if args.cuda else "cpu")

    # MNIST dataset normalized between [0, 1]
    try:
        with open(args.dataset, 'rb') as f:
            dataset_dict = pickle.load(f)
    except BaseException as e:
        print(str(e.__class__.__name__) + ": " + str(e))
        exit()

    X_train_labeled = dataset_dict["X_train_labeled"]
    y_train_labeled = dataset_dict["y_train_labeled"]
    X_train_unlabeled = dataset_dict["X_train_unlabeled"]
    y_train_unlabeled = dataset_dict["y_train_unlabeled"]
    X_val = dataset_dict["X_val"]
    y_val = dataset_dict["y_val"]
    X_test = dataset_dict["X_test"]
    y_test = dataset_dict["y_test"]

    X = np.vstack((X_train_labeled, X_train_unlabeled, X_val, X_test))
    y = np.hstack((y_train_labeled, y_train_unlabeled, y_val, y_test))

    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = MnistViVA(z_dim=args.z_dim, hidden_dim=args.hidden, zeta=args.zeta, rho=args.rho, device=device).to(device)
    if os.path.isfile(args.model):
        print("===> Loading Checkpoint to Resume '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        model.load_state_dict(checkpoint['model'])
        print("\t===> Loaded Checkpoint '{}' (epoch {})".format(args.model, checkpoint['epoch']))
    else:
        raise FileNotFoundError("\t====> no checkpoint found at '{}'".format(args.model))

    model.eval()

    embeddings = []
    labels = []
    with torch.no_grad():
        for data, target in tqdm(dataloader):
            data = data.to(device)
            labels.append(target)

            recon_loss_sum, y_logits, t_coords, latent_loss_sum, tsne_loss = model(data)
            embeddings.append(t_coords.detach().cpu())

    embeddings = torch.cat(embeddings, dim=0)
    embeddings = embeddings.numpy()

    labels = torch.cat(labels, dim=0)
    labels = labels.numpy()

    df_embedding = pd.DataFrame(data=embeddings, columns=['T1', 'T2'])
    df_embedding['Label'] = labels

    facets = sns.lmplot(
        x='T1', y='T2',
        data=df_embedding,
        fit_reg=False,
        legend=True,
        height=10,
        hue='Label',
        scatter_kws={"s": 10, "alpha": 0.3}
    )

    facets.set(xticks=[], yticks=[], xlabel='', ylabel='')
    sns.despine(left=True, bottom=True)

    os.makedirs(args.save, exist_ok=True)
    facets.savefig(os.path.join(args.save, "mnist_viva_embeddings.png"), bbox_inches='tight')
    plt.close()
    print("Embedding plot generated!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize ViVA Embeddings for MNIST')
    parser.add_argument('--dataset', default='Data/mnist_1000_labeled.pkl', type=str, help='dataset path')
    parser.add_argument('--model', default='Results/best_checkpoint.pth', type=str, help='path to a model checkpoint')
    args = parser.parse_args()

    # assuming that a config file was saved at the same path of the model checkpoint
    config_file = os.path.join(os.path.dirname(args.model), 'config.txt')
    with open(config_file, 'r') as f:
        args.__dict__.update(json.load(f))

    main(args)