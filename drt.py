from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import shutil
from utils import *
import torch.nn.functional as F
import os
import os.path as osp
import torch.utils.data
from domainNet_loader import DomainNetDataset
from model.resnet_dra import *
import time
import sys

parser = argparse.ArgumentParser(description='dynamic transfer for MSDA')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lbsm', type=float, default=.01, metavar='LBSM',
                    help='learning rate (default: 0.1)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--lr_f', type=float, default=0.01, metavar='LRF',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                    help='SGD momentum (default: 0.0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=10, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-layer', type=int, default=2, metavar='K',
                    help='how many layers for classifier')
parser.add_argument('--save_path', type=str, default='save/drt', metavar='B',
                    help='board dir')
parser.add_argument('--src_path', type=str, default='', metavar='B',
                    help='directory of source datasets')
parser.add_argument('--trg_path', type=str, default='', metavar='B',
                    help='directory of target datasets')
parser.add_argument('--val_path', type=str, default='', metavar='B',
                    help='directory of target datasets')
parser.add_argument('--root', type=str, default='', metavar='B',
                    help='data root dir')
parser.add_argument('--pretrain', type=str, default=None, metavar='B',
                    help='path to the pretrained model')
parser.add_argument('--eta', type=float, default=50.0, metavar='T',
                    help='eta for discrepancy')
parser.add_argument('--gamma', type=float, default=0.01, metavar='T',
                    help='gamma for entropy')
parser.add_argument('--lmbd', type=float, default=0.25, metavar='T',
                    help='lmbd for grad reverse')
parser.add_argument('--schedule', type=int, nargs='+', default=[30, 60, 90],
                    help='decrease learning rate at these epochs.')
parser.add_argument('--weight', type=str, default=None, metavar='B',
                    help='path to the model for evaluation')
parser.add_argument('--evaluate', default=False, action='store_true', 
                    help='using a separate teacher or not')
def print_options(save_path, opt):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    file_name = osp.join(save_path, 'options.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')

best_acc, start_epoch = 0, 0
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
src_path = args.src_path
trg_path = args.trg_path
val_path = args.val_path
num_layer = args.num_layer
batch_size = args.batch_size
save_path = args.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)
print_options(save_path, args)

data_transforms = {
    src_path: transforms.Compose([
        transforms.Scale(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),  
    trg_path: transforms.Compose([
        transforms.Scale(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    val_path: transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),    
}

dsets = dict()
src_dataset = DomainNetDataset(args.root, image_list=args.src_path, transform = data_transforms[src_path])
dsets[src_path] = torch.utils.data.DataLoader(src_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)

trg_dataset = DomainNetDataset(args.root, image_list=args.trg_path, transform = data_transforms[trg_path])
dsets[trg_path] = torch.utils.data.DataLoader(trg_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)

val_dataset = DomainNetDataset(args.root, image_list=args.val_path, transform = data_transforms[val_path])
dsets[val_path] = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)


dset_classes = 345 #src_dataset.num_classes
print ('num of classes: %d' %(dset_classes))
use_gpu = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

opt= args
print (args.pretrain)
G = resnet101_dy(pretrain=args.pretrain)
F1 = ResClassifier(num_classes=dset_classes, num_layer=num_layer, num_unit=2048)
F2 = ResClassifier(num_classes=dset_classes, num_layer=num_layer, num_unit=2048)
F1.apply(weights_init)
F2.apply(weights_init)
F1.set_lambda(args.lmbd)
F2.set_lambda(args.lmbd)

arch = open(os.path.join(save_path, 'arch.txt'), 'w')
print (G,file=arch)
arch.close()

lr = args.lr
if args.cuda:
    G.cuda()
    F1.cuda()
    F2.cuda()

optimizer_g = optim.SGD([
    {'params': get_params(G)},
    {'params': get_10xparams(G), 'lr':10*args.lr}
    ], momentum=args.momentum, lr=args.lr, weight_decay=0.0005)
optimizer_f = optim.SGD(list(F1.parameters())+list(F2.parameters()),momentum=0.9,lr=args.lr_f,weight_decay=0.0005)

if os.path.exists(osp.join(save_path, 'checkpoint.pth.tar')):
    print("=> loading checkpoint '{}'".format(osp.join(save_path, 'checkpoint.pth.tar')))
    checkpoint = torch.load(osp.join(save_path, 'checkpoint.pth.tar'))
    start_epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    G.load_state_dict(checkpoint['state_dict_G'])
    F1.load_state_dict(checkpoint['state_dict_F1'])
    F2.load_state_dict(checkpoint['state_dict_F2'])
    optimizer_g.load_state_dict(checkpoint['optimizer_G'])
    optimizer_f.load_state_dict(checkpoint['optimizer_F'])
    print("=> loaded checkpoint '{}' (epoch {})".format(osp.join(save_path, 'checkpoint.pth.tar'), checkpoint['epoch']))

iter_len_src = len(dsets[src_path]) - 1
iter_len_trg = len(dsets[trg_path]) - 1
def main():
    if args.evaluate:
        print("=> loading checkpoint '{}'".format(args.weight))
        model = torch.load(args.weight)
        G.load_state_dict(model['state_dict_G'])
        F1.load_state_dict(model['state_dict_F1'])
        F2.load_state_dict(model['state_dict_F2'])
        test(0)
        return
    global best_acc
    if args.lbsm == 0:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = LabelSmoothingLoss(smoothing=args.lbsm)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    entropy_accum = AverageMeter()
    loss1_accum = AverageMeter()
    loss2_accum = AverageMeter()
    loss_dis_accum = AverageMeter()
    eta, gamma = args.eta, args.gamma
    for ep in range(start_epoch, args.epochs):
        src_loader = iter(dsets[src_path])
        end = time.time()
        adjust_learning_rate(optimizer_g, optimizer_f, ep)
        for batch_idx in range(iter_len_src):
            G.train()
            F1.train()
            F2.train()
            if args.cuda:
                src_data, src_target = next(src_loader)
                if batch_idx % iter_len_trg == 0:
                    trg_loader = iter(dsets[trg_path])
                trg_data, trg_target = next(trg_loader)
                data_time.update(time.time() - end)
                src_data, src_target = src_data.cuda(), src_target.cuda()
                trg_data, trg_target = trg_data.cuda(), trg_target.cuda()

                data = torch.cat((src_data, trg_data),0)

                optimizer_g.zero_grad()
                optimizer_f.zero_grad()
                output = G(data)
                output1 = F1(output)
                output2 = F2(output)

                output_s1 = output1[:batch_size,:]
                output_s2 = output2[:batch_size,:]
                output_t1 = output1[batch_size:,:]
                output_t2 = output2[batch_size:,:]            

                output_t1 = F.softmax(output_t1)
                output_t2 = F.softmax(output_t2)

                entropy_loss = - torch.mean(torch.log(torch.mean(output_t1,0)+1e-6))
                entropy_loss -= torch.mean(torch.log(torch.mean(output_t2,0)+1e-6))

                loss1 = criterion(output_s1, src_target)
                loss2 = criterion(output_s2, src_target)

                loss1_accum.update(float(loss1.data.cpu()), output_s1.size(0))
                loss2_accum.update(float(loss2.data.cpu()), output_s2.size(0))
                entropy_accum.update(float(entropy_loss.data.cpu()), output_t1.size(0))

                output_t1_res = F1(output[batch_size:,:], reverse=True)
                output_t2_res = F2(output[batch_size:,:], reverse=True)   
                output_t1_res = F.softmax(output_t1_res)
                output_t2_res = F.softmax(output_t2_res)            

                loss_dis = torch.mean(torch.abs(output_t1_res-output_t2_res))
                loss_dis_accum.update(float(loss_dis.data.cpu()), output_t1_res.size(0))

                all_loss = loss1 + loss2 + gamma * entropy_loss - eta * loss_dis

                all_loss.backward()
                optimizer_g.step()
                optimizer_f.step()

            if batch_idx % args.log_interval == 0:
                log_str = ('Train Ep: {} [{}/{} ({:.0f}%)]\t'
                           'Time {batch_time_val:.3f} ({batch_time_avg:.3f})\t'
                           'Data {data_time_val:.3f} ({data_time_avg:.3f})\t'
                           'Loss1 {loss1_val:.6f} ({loss1_avg:.6f})\t'
                           'Loss2 {loss2_val:.6f} ({loss2_avg:.6f})\t'
                           'Entropy {entropy_loss_val:.6f} ({entropy_loss_avg:.6f})\t'
                           'Dis {loss_dis_val:.6f} ({loss_dis_avg:.6f})\t').format(
                           ep, batch_idx * batch_size, iter_len_src * batch_size,
                           100. * batch_idx / float(iter_len_src),
                           batch_time_val=batch_time.val, batch_time_avg=batch_time.avg, 
                           data_time_val=data_time.val, data_time_avg=data_time.avg,
                           loss1_val=loss1_accum.val, loss1_avg=loss1_accum.avg,
                           loss2_val=loss2_accum.val, loss2_avg=loss2_accum.avg,
                           entropy_loss_val=entropy_accum.val, entropy_loss_avg=entropy_accum.avg, 
                           loss_dis_val=loss_dis_accum.val,loss_dis_avg=loss_dis_accum.avg)
                print(log_str)
                with open(osp.join(save_path, 'train_log.txt'), 'a') as fp:
                    fp.write(log_str+'\n')
            batch_time.update(time.time() - end)
            end = time.time()
        acc = test(ep)
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint({
            'epoch': ep + 1,
            'state_dict_G': G.state_dict(),
            'state_dict_F1': F1.state_dict(),
            'state_dict_F2': F2.state_dict(),
            'best_acc': best_acc,
            'optimizer_G': optimizer_g.state_dict(),
            'optimizer_F': optimizer_f.state_dict(),
            }, is_best, ep, save=save_path, filename='checkpoint.pth.tar')



def test(epoch):
    G.eval()
    F1.eval()
    F2.eval()
    test_loss = 0
    correct = 0
    correct2 = 0
    correct_avg = 0
    size = 0
    val = False
    val_loader = dsets[val_path]
    for batch_idx, input in enumerate(val_loader):
        #if batch_idx*batch_size > 5000:
        #    break
        data, target = input
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output = G(data)
            output1 = F1(output)
            output2 = F2(output)
        test_loss += float(F.nll_loss(output1, target).data.cpu())
        pred = output1.data.max(1)[1] # get the index of the max log-probability
        correct += float(pred.eq(target.data).cpu().sum())
        pred2 = output2.data.max(1)[1] # get the index of the max log-probability
        k = target.data.size()[0]
        correct2 += float(pred2.eq(target.data).cpu().sum())
        size += k
        pred_avg = pred
        for i in range(k):
            if output1[i][pred[i]] < output2[i][pred2[i]]:
                pred_avg[i] = pred2[i]
        correct_avg += float(pred_avg.eq(target.data).cpu().sum())
        #break

    test_loss = test_loss
    test_loss /= len(val_loader) # loss function already averages over batch size
    acc1, acc2, acc = 100. * float(correct) / float(size), 100. * float(correct2) / float(size), 100. * float(correct_avg) / float(size)
    log_str = 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%) ({:.2f}%) ({:.2f}%)\n'.format(
        test_loss, correct, size, acc1, acc2, acc)
    print('\n'+log_str)
    if args.evaluate:
        return 
    with open(osp.join(save_path, 'test_log.txt'), 'a') as fp:
        fp.write(log_str)
    return acc


def save_checkpoint(state, is_best, epoch, save = '', filename='checkpoint.pth.tar'):
    torch.save(state, osp.join(save, filename))
    if is_best:
        shutil.copyfile(osp.join(save, filename), osp.join(save, 'model_best.pth.tar'))
    if (epoch + 1) % 5 == 0:
        shutil.copyfile(osp.join(save, filename), osp.join(save, 'checkpoint_ep{}.pth.tar'.format(epoch + 1)))


class AverageMeter(object):
    """Computes and stores the average and current value"""
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

def adjust_learning_rate(optimizer_g, optimizer_f, ep):
    count = sum([1 for s in args.schedule if s <= ep])
    optimizer_g.param_groups[0]['lr'] = args.lr * pow(0.1, count)
    optimizer_g.param_groups[1]['lr'] = 10*args.lr * pow(0.1, count)
    optimizer_f.param_groups[0]['lr'] = args.lr_f * pow(0.1, count)

class LabelSmoothingLoss(nn.Module):

    def __init__(self, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = input.log_softmax(dim=-1)
        weight = input.new_ones(input.size()) * \
            self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

if __name__ == '__main__':
    main()
