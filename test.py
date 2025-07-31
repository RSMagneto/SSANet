import pdb
import re
from torch.autograd import Variable
import model_wheat as model
import torch
import functions
import numpy as np
import os
from skimage import io
import argparse
import scipy.io as sio
import dataloader
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', help='input image dir', default='/mnt/wheat/dataset/wheat/test/')
    parser.add_argument('--modelpath',  help='output model dir', default='/mnt/wheat/output/best.pth')
    parser.add_argument('--device', default=torch.device('cuda:1'))
    parser.add_argument('--classes', default=6)
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--batchSize', default=1)

    opt = parser.parse_args()
    Net = model.SSANet(opt.classes, 64, 65, 64).to(opt.device)
    Net.eval()
    modelname = opt.modelpath
    Net.load_state_dict(torch.load(modelname))
    predicted = []
    labl = []
    test_data_infos = dataloader.get_test_DataInfo(opt.test_dir, opt.classes)
    test_set = dataloader.get_test_set(test_data_infos)
    test_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    test_txt = '/mnt/wheat_classification_six/123_new'
    for batch_idx, (imgs, labels) in enumerate(test_loader):
        if torch.cuda.is_available():
            imgs, labels = imgs.to(opt.device), labels.to(opt.device)
            imgs = Variable(imgs.to(torch.float32))
            labels = Variable(labels.to(torch.float32))
        out,_ = Net(imgs)
        if isinstance(out,tuple):
            out = out[0]
        out = out.data.cpu().numpy()
        labels = labels.data.cpu().numpy()
        predicted.append(out)
        labl.append(labels)
    outp = np.array(predicted)
    lab = np.array(labl)

    y_pred_test = np.argmax(outp, axis=2)
    kappa = cohen_kappa_score(lab, y_pred_test)
    oa = accuracy_score(lab, y_pred_test)
    confusion = confusion_matrix(lab, y_pred_test)
    each_acc, aa = functions.AA_andEachClassAccuracy(confusion)
    # each_acc, aa = functions.calculate_OA_AA(confusion)

    print('OverallAccuracy  = ', format(oa * 100, ".2f") + ' %')
    print('Average Accuracy = ', format(aa * 100, ".2f") + ' %')
    print('Kappa            = ', format(kappa * 100, ".2f") + ' %')
    classification, confusion, evaluate, each_acc = functions.reports(np.argmax(outp, axis=2), lab)
    print(confusion)
    print('OA, AA, kappa:', evaluate)
    print('each-acc：', each_acc)
    outputt = "OverallAccuracy: %f, AverageAccuracy: %f, Kappa: %f, \n each_acc: %s, \n confusion: \n %s" % (oa, aa, kappa, each_acc, confusion)
    with open(test_txt, "a+") as f:
        f.write(outputt + '\n')
        f.close

if __name__ == "__main__":
    main()
