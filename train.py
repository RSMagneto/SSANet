import argparse
import pdb
import scipy.io as sio
import model_wheat as model
import torch
import torch.nn as nn
import functions
import time
import os
import copy
import random
import dataloader
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', help='input image dir', default='dataset/wheat/train/')
    parser.add_argument('--val_dir', help='input image dir', default='dataset/wheat/val/')
    parser.add_argument('--test_dir', help='input image dir', default='dataset/wheat/test/')
    parser.add_argument('--outputs_dir', help='output model dir', default='/mnt/wheat_new/output/mdoel_wheat')
    parser.add_argument('--batchSize', default=16)
    parser.add_argument('--classes', default=6)
    parser.add_argument('--testBatchSize', default=64)
    parser.add_argument('--epoch', default=100)
    parser.add_argument('--result', default='./result/', type=str, help='model and figure')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--not_cuda', action='store_true', help='disables cuda', default=1)
    parser.add_argument('--device', default=torch.device('cuda:0'))
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--lr', type=float, default=1e-3, help='G‘s learning rate')
    parser.add_argument('--gamma', type=float, default=0.1, help='scheduler gamma')

    opt = parser.parse_args()
    train_data_infos = dataloader.get_train_DataInfo(opt.train_dir, opt.classes)
    val_data_infos = dataloader.get_val_DataInfo(opt.val_dir, opt.classes)
    train_set = dataloader.get_training_set(train_data_infos)
    val_set = dataloader.get_val_set(val_data_infos)
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=True)
    Net = model.SSANet(opt.classes, 64, 65, 64).to(opt.device)
    for module in Net.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    optimizer = torch.optim.Adam(Net.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[50], gamma=opt.gamma)
    loss = nn.CrossEntropyLoss().to(opt.device)
    best_acc = -1
    loss_list = []
    train_acc_list = []
    val_acc_list = []
    best_loss = 0.
    best_train = 0.
    best_epoch = 0
    best_model = None
    best_optimizer = None
    epoch = 0
    time_start = time.time()

    for i in range(opt.epoch):
        # train
        Net.train()
        accs = np.ones((len(train_loader))) * -1000.0
        losses = np.ones((len(train_loader))) * -1000.0
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                imgs, labels = imgs.to(opt.device), labels.to(opt.device)
                imgs = Variable(imgs.to(torch.float32))
                labels = Variable(labels.to(torch.float32))
            labels = torch.tensor(labels, dtype=torch.long)
            out, feature_map = Net(imgs)
            out = Net(imgs)
            outLoss = loss(out, labels)
            losses[batch_idx] = outLoss.item()
            accs[batch_idx] = functions.accuracy(out.data, labels.data)[0].item()
            optimizer.zero_grad()
            outLoss.backward(retain_graph=True)
            optimizer.step()
        train_loss = np.average(losses)
        train_acc = np.average(accs)
        print("epoch:", i, "train loss:", train_loss, "train acc", train_acc)
        torch.save(Net.state_dict(), os.path.join(opt.outputs_dir, 'epoch_{}.pth'.format(i)))

        Net.eval()
        accs = np.ones((len(val_loader))) * -1000.0
        losses = np.ones((len(val_loader))) * -1000.0
        with torch.no_grad():
            for batch_idx, (img, label) in enumerate(val_loader):
                if torch.cuda.is_available():
                    img, label = img.to(opt.device), label.to(opt.device)
                    img = Variable(img.to(torch.float32))
                    label = Variable(label.to(torch.float32))
                label = torch.tensor(label, dtype=torch.long)
                output = Net(img)
                loss1 = loss(output, label)
                losses[batch_idx] = loss1.item()
                accs[batch_idx] = functions.accuracy(output.data, label.data, topk=(1,))[0].item()
            val_loss = np.average(losses)
            val_acc = np.average(accs)
            print("epoch:", i, "val loss:", val_loss, "val acc", val_acc)
            if val_acc > best_acc:
                best_acc = val_acc
                best_train = train_acc
                best_loss = train_loss
                best_epoch = epoch + 1
                best_model = copy.deepcopy(Net)
            loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)
            best_records = {'epoch': best_epoch, 'loss': best_loss,
                            'train_acc': best_train, 'val_acc': best_acc}
            time_end = time.time()
            train_time = time_end - time_start
            mean_time = train_time / opt.epoch
            torch.save(Net.state_dict(), os.path.join(opt.outputs_dir, 'best.pth'))

            train_end_time = ( time_end - time_start) / 3600
            print(f'train all time:{train_end_time:.4f} hour')

            graph_data = {
                'loss': loss_list,
                'train accuracy': train_acc_list,
                'val accuracy': val_acc_list,
            }
        scheduler.step()