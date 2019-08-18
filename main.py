import argparse
import random
import shutil
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from functions.logger import Logger
from functions.get_data.mat2npy import mat2npy
from functions.get_data.prepare_local_feature import prepare_local_feature
from functions.get_data.prepare_pairwise_feature import prepare_hist_pairwise_feature
from functions.get_data.dataset import MyDataSet
from functions.model import ClassifierNetwork
from functions.train import train, validate
from functions.utils import save_checkpoint
from functions.test import test

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="rotated_vertical_classifier") # if model changes, bin_num and edge_num need to be changed
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--gpu', default=1, type=int,help='GPU id to use.')
parser.add_argument('--edge_num', default=50, type=int) # each neuron's edge
parser.add_argument('--bin_num', default=36, type=int)
parser.add_argument('--patch_size', default=64, type=int)
parser.add_argument('--classes', default=107, type=int)
parser.add_argument('--resume', default='', type=str, metavar='PATH')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N')
parser.add_argument('--print-freq', '-p', default=30, type=int,metavar='N')
parser.add_argument('--patch_nearest_num',default=50,type=int, help='used to determine the direction in PCA')
parser.add_argument('--head_neuron',default=77,type=int)
parser.add_argument('--tail_neuron',default=1,type=int)
parser.add_argument('--bi_stochastic_iter',default=2000,type=int)

args = parser.parse_args()


random.seed(args.seed)
torch.manual_seed(args.seed)
cudnn.deterministic = True

NPY_PATH = './data/npy_files/'
FEATURE_PATH = './data/ordered_dataset/'

def main(args):
    print(vars(args))
    print('model is', args.model)

    mat2npy(save_dir=NPY_PATH)

    prepare_local_feature(patch_size=args.patch_size, num_nearest_neurons=args.patch_nearest_num,
                          head_neuron=args.head_neuron, tail_neuron=args.tail_neuron, data_root=NPY_PATH, save_root=FEATURE_PATH)

    prepare_hist_pairwise_feature(num_bins=args.bin_num, num_edge=args.edge_num,
                                  head_neuron=args.head_neuron, tail_neuron=args.tail_neuron,
                                  file_root=NPY_PATH, save_root=FEATURE_PATH)


    train_loader = torch.utils.data.DataLoader(MyDataSet(root=FEATURE_PATH, mode='train', patch_size=args.patch_size, num_bins=args.bin_num,
                                                         num_edges=args.edge_num,patch_nearest = args.patch_nearest_num),
                                                         batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(MyDataSet(root=FEATURE_PATH, mode='val', patch_size=args.patch_size, num_bins=args.bin_num,
                                                       num_edges=args.edge_num, patch_nearest = args.patch_nearest_num),
                                                       batch_size=args.batch_size, shuffle=True)

    model = ClassifierNetwork(bin_num=args.bin_num, edge_num=args.edge_num).cuda(args.gpu)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    train_logger_path = './logs/{}_bins_{}_edges_{}_lr_{}_wd{}/train'.format(
            args.model,args.bin_num, args.edge_num, args.lr, args.weight_decay)
    val_logger_path = './logs/{}_bins_{}_edges_{}_lr_{}_wd_{}/val'.format(
        args.model,args.bin_num, args.edge_num, args.lr, args.weight_decay)
    if os.path.exists(train_logger_path):
        shutil.rmtree(train_logger_path)
    if os.path.exists(val_logger_path):
        shutil.rmtree(val_logger_path)
    logger_train = Logger(train_logger_path)
    logger_val = Logger(val_logger_path)

    best_acc = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            best_acc = checkpoint['best_acc']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    for epoch in range(args.start_epoch,args.epochs):
        # print('learing rate is {}\n'.format(optimizer.param_groups[0]['lr']))

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, logger_train, args)

        # evaluate for one epoch
        val_loss, val_acc = validate(val_loader, model, criterion, epoch, logger_val, args)

        # adjust for proper learning rate
        # scheduler.step(val_loss)

        # save checkpoint
        model_save_dir = './trained_model/'
        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)
        model_save_path = os.path.join(model_save_dir,
                                 '{}_bins_{}_edges_{}_epochs_{}.pth.tar'.format(args.model, args.bin_num, args.edge_num,
                                                                                args.epochs))

        # save best_acc checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        if is_best:
            best_epoch = epoch
            save_checkpoint({'epoch': epoch + 1, 'best_acc': best_acc,'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                            model_save_path)

    print('best top1 accuracy is', best_acc)
    print('best top1 accuracy epoch is', best_epoch)

    print('-' * 20)
    print('Testing')
    test(args=args, model_save_dir='./trained_model')


main(args)
