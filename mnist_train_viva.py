import argparse
import json
import os
import pickle
import random
import sys

from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, TensorDataset

from torch.optim.adam import Adam

from utils import one_hot, save_checkpoint, AverageMeter, CosineAnnealing
from models import MnistViVA


def main(args):
    print('===> Configuration')
    print(args)

    os.makedirs(args.save, exist_ok=True)
    with open(os.path.join(args.save, "config.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    cudnn.benchmark = True if args.cuda else False
    device = torch.device("cuda" if args.cuda else "cpu")

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

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

    labeled_dataset = TensorDataset(torch.from_numpy(X_train_labeled).float(), torch.from_numpy(y_train_labeled).long())
    unlabeled_dataset = TensorDataset(torch.from_numpy(X_train_unlabeled).float(), torch.from_numpy(y_train_unlabeled).long())
    val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())

    NUM_SAMPLES = len(labeled_dataset) + len(unlabeled_dataset)
    NUM_LABELED = len(labeled_dataset)

    labeled_dataloader = DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=False)
    unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=False)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    alpha = args.eta * NUM_SAMPLES / NUM_LABELED
    tau = CosineAnnealing(start=1.0, stop=0.5, t_max=args.tw, mode='down')

    model = MnistViVA(z_dim=args.z_dim, hidden_dim=args.hidden, zeta=args.zeta, rho=args.rho, device=device).to(device)
    optimizer = Adam(model.parameters())

    best_val_epoch = 0
    best_val_loss = sys.float_info.max
    best_val_acc = 0.0
    test_acc = 0.0
    early_stop_counter = 0

    if args.resume:
        if os.path.isfile(args.resume):
            print("===> Loading Checkpoint to Resume '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            best_val_epoch = checkpoint['best_epoch']
            best_val_loss = checkpoint['best_val_loss']
            best_val_acc = checkpoint['best_val_acc']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("\t===> Loaded Checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            raise FileNotFoundError("\t====> no checkpoint found at '{}'".format(args.resume))

    n_batches = len(labeled_dataloader) + len(unlabeled_dataloader)
    n_unlabeled_per_labeled = len(unlabeled_dataloader) // len(labeled_dataloader) + 1

    with tqdm(range(args.start_epoch, args.epochs), desc="Epochs") as nested:
        for epoch in nested:

            # Train
            model.train()
            train_recon_loss = AverageMeter('Train_Recon_Loss')
            train_latent_loss = AverageMeter('Train_Latent_Loss')
            train_label_loss = AverageMeter('Train_Label_Loss')
            train_tsne_loss = AverageMeter('Train_tSNE_Loss')
            train_total_loss = AverageMeter('Train_Total_Loss')
            train_accuracy = AverageMeter('Train_Accuracy')

            labeled_iter = iter(labeled_dataloader)
            unlabeled_iter = iter(unlabeled_dataloader)

            for batch_idx in range(n_batches):

                is_supervised = batch_idx % n_unlabeled_per_labeled == 0
                # get batch from respective dataloader
                if is_supervised:
                    try:
                        data, target = next(labeled_iter)
                        data = data.to(device)
                        target = target.to(device)
                        one_hot_target = one_hot(target, 10)
                    except StopIteration:
                        data, target = next(unlabeled_iter)
                        data = data.to(device)
                        target = target.to(device)
                        one_hot_target = None
                else:
                    data, target = next(unlabeled_iter)
                    data = data.to(device)
                    target = target.to(device)
                    one_hot_target = None

                model.zero_grad()

                recon_loss_sum, y_logits, t_coords, latent_loss_sum, tsne_loss = model(data, one_hot_target, tau.step())
                recon_loss = recon_loss_sum / data.size(0)
                label_loss = F.cross_entropy(y_logits, target, reduction='mean')
                latent_loss = latent_loss_sum / data.size(0)

                # Full loss
                total_loss = recon_loss + latent_loss + args.gamma * tsne_loss
                if is_supervised and one_hot_target is not None:
                    total_loss += alpha * label_loss

                assert not np.isnan(total_loss.item()), 'Model diverged with loss = NaN'

                train_recon_loss.update(recon_loss.item())
                train_latent_loss.update(latent_loss.item())
                train_label_loss.update(label_loss.item())
                train_tsne_loss.update(tsne_loss.item())
                train_total_loss.update(total_loss.item())

                total_loss.backward()
                optimizer.step()

                pred = y_logits.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                train_correct = pred.eq(target.view_as(pred)).sum().item()
                train_accuracy.update(train_correct / data.size(0), data.size(0))

                if batch_idx % args.log_interval == 0:
                    tqdm.write('Train Epoch: {} [{}/{} ({:.0f}%)]\t Recon: {:.6f} Latent: {:.6f} t-SNE: {:.6f} Accuracy {:.4f} T {:.6f}'.format(
                        epoch, batch_idx, n_batches, 100. * batch_idx / n_batches, train_recon_loss.avg, train_latent_loss.avg, train_tsne_loss.avg, train_accuracy.avg, tau.value))

            tqdm.write('====> Epoch: {} Average train loss - Recon {:.3f} Latent {:.3f} t-SNE {:.6f} Label {:.6f} Accuracy {:.4f}'.format(epoch, train_recon_loss.avg, train_latent_loss.avg, train_tsne_loss.avg, train_label_loss.avg, train_accuracy.avg))

            # Validation
            model.eval()

            val_recon_loss = AverageMeter('Val_Recon_Loss')
            val_latent_loss = AverageMeter('Val_Latent_Loss')
            val_label_loss = AverageMeter('Val_Label_Loss')
            val_tsne_loss = AverageMeter('Val_tSNE_Loss')
            val_total_loss = AverageMeter('Val_Total_Loss')
            val_accuracy = AverageMeter('Val_Accuracy')

            with torch.no_grad():
                for i, (data, target) in enumerate(val_loader):
                    data = data.to(device)
                    target = target.to(device)

                    recon_loss_sum, y_logits, t_coords, latent_loss_sum, tsne_loss = model(data, temperature=tau.value)

                    recon_loss = recon_loss_sum / data.size(0)
                    label_loss = F.cross_entropy(y_logits, target, reduction='mean')
                    latent_loss = latent_loss_sum / data.size(0)

                    # Full loss
                    total_loss = recon_loss + latent_loss + args.gamma * tsne_loss + alpha * label_loss

                    val_recon_loss.update(recon_loss.item())
                    val_latent_loss.update(latent_loss.item())
                    val_label_loss.update(label_loss.item())
                    val_tsne_loss.update(tsne_loss.item())
                    val_total_loss.update(total_loss.item())

                    pred = y_logits.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    val_correct = pred.eq(target.view_as(pred)).sum().item()
                    val_accuracy.update(val_correct / data.size(0), data.size(0))

            tqdm.write('\t Validation loss - Recon {:.3f} Latent {:.3f} t-SNE {:.6f} Label: {:.6f} Accuracy {:.4f}'.format(val_recon_loss.avg, val_latent_loss.avg, val_tsne_loss.avg, val_label_loss.avg, val_accuracy.avg))

            is_best = val_accuracy.avg > best_val_acc
            if is_best:
                early_stop_counter = 0
                best_val_epoch = epoch
                best_val_loss = val_total_loss.avg
                best_val_acc = val_accuracy.avg

                test_accuracy = AverageMeter('Test_Accuracy')
                with torch.no_grad():
                    for i, (data, target) in enumerate(test_loader):
                        data = data.to(device)
                        target = target.to(device)

                        _, y_logits, _, _, _ = model(data, temperature=tau.value)

                        pred = y_logits.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                        test_correct = pred.eq(target.view_as(pred)).sum().item()
                        test_accuracy.update(test_correct / data.size(0), data.size(0))

                test_acc = test_accuracy.avg
                tqdm.write('\t Test Accuracy {:.4f}'.format(test_acc))
                with open(os.path.join(args.save, 'train_result.txt'), 'w') as f:
                    f.write('Best Validation Epoch: {}\n'.format(epoch))
                    f.write('Train Recon Loss: {}\n'.format(train_recon_loss.avg))
                    f.write('Train Latent Loss: {}\n'.format(train_latent_loss.avg))
                    f.write('Train tSNE Loss: {}\n'.format(train_tsne_loss.avg))
                    f.write('Train Label Loss: {}\n'.format(train_label_loss.avg))
                    f.write('Train Total Loss: {}\n'.format(train_total_loss.avg))
                    f.write('Train Accuracy: {}\n'.format(train_accuracy.avg))
                    f.write('Val Recon Loss: {}\n'.format(val_recon_loss.avg))
                    f.write('Val Latent Loss: {}\n'.format(val_latent_loss.avg))
                    f.write('Val tSNE Loss: {}\n'.format(val_tsne_loss.avg))
                    f.write('Val Label Loss: {}\n'.format(val_label_loss.avg))
                    f.write('Val Total Loss: {}\n'.format(val_total_loss.avg))
                    f.write('Val Accuracy: {}\n'.format(val_accuracy.avg))
                    f.write('Test Accuracy: {}\n'.format(test_acc))
            else:
                early_stop_counter += 1

            save_checkpoint({
                'epoch': epoch,
                'best_epoch': best_val_epoch,
                'best_val_loss': best_val_loss,
                'best_val_accuracy': best_val_acc,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best, filename=os.path.join(args.save, 'checkpoint.pth'))

            if args.early_stop > 0 and early_stop_counter == args.early_stop:
                tqdm.write("Early Stop with no improvement: epoch {}".format(epoch))
                break

    print("Training is Completed!")
    print("Best Val Acc: {:.4f} Test Acc: {:.4f}".format(best_val_acc, test_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ViVA on MNIST')

    parser.add_argument('--dataset', default='Data/mnist_1000_labeled.pkl', type=str, help='dataset path')

    parser.add_argument('--hidden', type=int, default=400, help='hidden dim (default: 400)')
    parser.add_argument('--z-dim', type=int, default=20, help='z dim (default: 20)')

    parser.add_argument('--eta', type=float, default=10.0, help='alpha scale (classification loss weight) (default: 10)')
    parser.add_argument('--gamma', type=float, default=100.0, help='gamma (tsne loss weight) (default: 100)')
    parser.add_argument('--zeta', type=float, default=10.0, help='zeta (sigma multiplier) (default: 10)')
    parser.add_argument('--rho', type=float, default=0.5, help='rho (class penalty) (default: 0.5)')
    parser.add_argument('--tw', type=int, default=1000, help='Gumbel-Softmax temperature annealing period (default: 1000)')

    parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train (default: 1000)')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--early-stop', default=0, type=int, help='num epochs of no improvement required to terminate (default: 0 means not using it)')

    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers (default: 4)')

    parser.add_argument('--save', default='Results', type=str, help='save path')
    parser.add_argument('--resume', default='', type=str, help='path to a checkpoint (default: none)')

    args = parser.parse_args()

    if args.save == '':
        args.save = './'
    args.save = os.path.abspath(args.save)

    main(args)
