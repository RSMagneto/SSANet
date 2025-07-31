import skimage.io as skimage
import torch
import numpy as np
from numpy import *
import pdb
from sewar.full_ref import sam
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
import cv2
from skimage import util
import torch.nn.functional as F

def test_matRead(data,opt):
    data = data[None, :, :, :]
    # data = data.transpose(0, 3, 1, 2)/32701.#WSDC
    data = data.transpose(0, 3, 1, 2) / 64000. #CAVE
    # data = data.transpose(0, 3, 1, 2) / 0.07
    # data = data.transpose(0, 3, 1, 2)/8000. #PAVIA
    data = torch.from_numpy(data)
    data = data.type(torch.cuda.FloatTensor)
    data = data.to(opt.device)
    # data=(data-0.5)*2
    # data=data.clamp(-1,1)   #归一化
    return data

def getBatch(hsBatch, msBatch, hrhsBatch, bs):
    N = hrhsBatch.shape[0]
    batchIndex = np.random.randint(0, N, size=bs)
    hrmsBatch = msBatch[batchIndex, :, :, :]
    gtBatch = hrhsBatch[batchIndex, :, :, :]
    lrhsBatch = hsBatch[batchIndex, :, :, :]
    return lrhsBatch, hrmsBatch, gtBatch

def getTest(hrms, label, gt_data, lrhs):
    N = gt_data.shape[0]
    batchIndex = np.random.randint(0, N, size=1)
    hrmsBatch = torch.linalg.invhrms[batchIndex, :, :, :]
    labelBatch = label[batchIndex, :, :]
    gtBatch = gt_data[batchIndex, :, :, :]
    lrhsBatch = lrhs[batchIndex, :, :, :]
    return hrmsBatch, labelBatch, gtBatch, lrhsBatch

def convert_image_np(inp,opt):
    inp = inp[-1, :, :, :]
    inp = inp.to(torch.device('cpu'))
    inp = inp.numpy().transpose((1, 2, 0))
    # inp = np.clip(inp/ 2 + 0.5,0,1)
    inp = np.clip(inp, 0, 1)
    inp = (inp) * 64000.
    return inp

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def SAM(sr_img, hr_img):
    sr_img = sr_img.to(torch.device('cpu'))
    sr_img = sr_img.numpy()
    sr_img = sr_img[-1, :, :, :]
    hr_img = hr_img.to(torch.device('cpu'))
    hr_img = hr_img.numpy()
    hr_img = hr_img[-1, :, :, :]
    sam_value = sam(sr_img*1.0, hr_img*1.0)
    return sam_value

def normalize(data):
    h, w, c = data.shape
    data = data.reshape((h * w, c))
    data -= np.min(data, axis=0)
    data /= np.max(data, axis=0)
    data = data.reshape((h, w, c))
    return data

def calc_psnr(img_tgt,img_fus):
    img_tgt = img_tgt.reshape(-1, img_tgt.shape[0])
    img_fus = img_fus.reshape(-1,img_fus.shape[0])
    mse = torch.mean(torch.square(img_tgt-img_fus))
    img_max = torch.max(img_tgt)
    psnr = 10.0 * torch.log10(img_max**2/mse)
    return psnr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)  # 输出对角线上的元素
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))  # list_diag/list_raw_sum C内核
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def calculate_OA_AA(y_true,y_pred):
    cm = confusion_matrix(y_true, y_pred)
    OA =np.trace(cm) / np.sum(cm)
    class_acc =[]
    for i in range(cm.shape[0]):
        correct = cm[i,i]
        total = np.sum(cm[i, :])
        class_acc.append(correct / total if total != 0 else 0)
        AA =np.mean(class_acc)
        return OA,AA

def adjust_learning_rate(optimizer, epoch, lr):
    lr = lr * (0.1 ** (epoch // 75))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def reports(y_pred, y_test):
    classification = classification_report(y_test, y_pred)
    oa = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred)
    evaluate = [oa, aa, kappa]
    each_acc = list(np.round(each_acc * 100, 2))  # 百分制，取两位有效
    return classification, confusion, evaluate, each_acc


def BGR2GRAY(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()
    out = 0.2126 * r + 0.7152 * g + 0.0722 * b
    out = out.astype(np.uint8)
    return out

def saliency(data):
    data = data.astype(np.float32)
    out = BGR2GRAY(data*255)
    ret2, th2 = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img1 = util.invert(th2)
    return img1


def PatchNCELoss(feat_q, feat_k, batch_size=16, nce_T=0.07):
    batch_size = batch_size
    nce_T = nce_T
    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
    mask_dtype = torch.bool

    num_patches = feat_q.shape[0]
    dim = feat_q.shape[1]
    feat_k = feat_k.detach()

    l_pos = torch.bmm(feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
    l_pos = l_pos.view(num_patches, 1)

    # reshape features to batch size
    feat_q = feat_q.view(batch_size, -1, dim)
    feat_k = feat_k.view(batch_size, -1, dim)
    npatches = feat_q.size(1)
    l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))
    diagonal = torch.eye(npatches, device=feat_q.device, dtype=mask_dtype)[None, :, :]
    l_neg_curbatch.masked_fill_(diagonal, -10.0)
    l_neg = l_neg_curbatch.view(-1, npatches)

    out = torch.cat((l_pos, l_neg), dim=1) / nce_T

    loss = cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                               device=feat_q.device))

    return loss
