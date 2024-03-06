import argparse
import os
import numpy as np
import pandas as pd
import torchvision
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
from train_generators import GeneratorResnet
import random

seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

parser = argparse.ArgumentParser(description='Training EGS_TSSA for generating sparse adversarial examples')
parser.add_argument('--train_dir', default='../share_dataset/ILSVRC-2012/ILSVRC2012_img_train', help='path to imagenet training set')
parser.add_argument('--model_type', type=str, default='res50', help='Model against GAN is trained: incv3, res50')
parser.add_argument('--eps', type=int, default=10, help='Perturbation Budget')
parser.add_argument('--target', type=int, default=-1, help='-1 if untargeted')
parser.add_argument('--batch_size', type=int, default=8, help='Number of trainig samples/batch')
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=2.25e-5, help='Initial learning rate for adam')
parser.add_argument('--checkpoint', type=str, default='', help='path to checkpoint')
parser.add_argument('--tk', type=float, default=0.6, help='path to checkpoint')

lam_1 = 0.00
lam_2 = 0.00001

# lam_1 = 0.0001
# lam_2 = 0.0003

args = parser.parse_args()
eps = args.eps
print(args)

TK = True
if TK == True:
    tk = args.tk
else:
    choose = [0.,0.5]

epochs = args.epochs
# GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

back_fea = torch.tensor([]).to(device)
back_grad = torch.tensor([]).to(device)


# 获取梯度的函数
def backward_hook(module, grad_in, grad_out):
    global back_grad
    back_grad = grad_out[0].clone().detach()


# 获取特征层的函数
def forward_hook(module, input, output):
    global back_fea
    back_fea = output.detach()


# Model
if args.model_type == 'incv3':
    model = torchvision.models.inception_v3(pretrained=True)
    model.Mixed_7c.register_forward_hook(forward_hook)
    model.Mixed_7c.register_full_backward_hook(backward_hook)
elif args.model_type == 'res50':
    model = torchvision.models.resnet50(pretrained=True)
    model.layer4[-1].register_forward_hook(forward_hook)
    model.layer4[-1].register_full_backward_hook(backward_hook)

model = model.to(device)
model.eval()

# Input dimensions
if args.model_type in ['res50']:
    scale_size = 256
    img_size = 224
    filterSize = 8
    stride = 8
else:
    scale_size = 300
    img_size = 299
    filterSize = 13
    stride = 13
# x_box
P = np.floor((img_size - filterSize) / stride) + 1
P = P.astype(np.int32)
Q = P
index = np.ones([P * Q, filterSize * filterSize], dtype=int)
tmpidx = 0
for q in range(Q):
    plus1 = q * stride * img_size
    for p in range(P):
        plus2 = p * stride
        index_ = np.array([], dtype=int)
        for i in range(filterSize):
            plus = i * img_size + plus1 + plus2
            index_ = np.append(index_, np.arange(plus, plus + filterSize, dtype=int))
        index[tmpidx] = index_
        tmpidx += 1
index = torch.LongTensor(np.tile(index, (args.batch_size, 1, 1))).to(device)

# Generator
if args.model_type == 'incv3':
    netG = GeneratorResnet(inception=True, eps=eps / 255.)
else:
    netG = GeneratorResnet(eps=eps / 255.)
if args.checkpoint != '':
    netG.load_state_dict(torch.load(args.checkpoint,map_location='cuda:0'))
netG.to(device)

# Optimizer
optimG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))


def trans_incep(x):
    x = F.interpolate(x, size=(299,299), mode='bilinear', align_corners=False)
    return x

# Data
data_transform = transforms.Compose([
    transforms.Resize(scale_size, antialias=True),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
])

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]

    return t


train_set = datasets.ImageFolder(args.train_dir, data_transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                           pin_memory=True)
train_size = len(train_set)
print('Training data size:', train_size)

# Loss
def CWLoss(logits, target, kappa=-0., tar=False):
    target = torch.ones(logits.size(0)).to(device).type(torch.cuda.FloatTensor).mul(target.float())
    target_one_hot = Variable(torch.eye(1000).type(torch.cuda.FloatTensor)[target.long()].to(device))

    real = torch.sum(target_one_hot * logits, 1)
    other = torch.max((1 - target_one_hot) * logits - (target_one_hot * 10000), 1)[0]
    kappa = torch.zeros_like(other).fill_(kappa)

    if tar:
        return torch.sum(torch.max(other - real, kappa))
    else:
        return torch.sum(torch.max(real - other, kappa))


criterion = CWLoss


def grad_topk(grad, index, filterSize, Tk):
    k = int(((img_size / filterSize) ** 2) * Tk)
    box_size = filterSize * filterSize
    for i in range(len(grad)):
        tmp = torch.take(grad[i], index[i])
        norm_tmp = torch.norm(tmp, dim=-1)
        g_topk = torch.topk(norm_tmp, k=k, dim=-1)
        top = g_topk.values.max() + 1
        norm_tmp_k = norm_tmp.put_(g_topk.indices, torch.FloatTensor([top] * k).to(device))
        norm_tmp_k = torch.where(norm_tmp_k == top, 1., 0.)
        tmp_bi = torch.as_tensor(norm_tmp_k.repeat_interleave(box_size)) * 1.0
        grad[i] = grad[i].put_(index[i], tmp_bi)
    return grad

def grad_choose(grad, index, filterSize, choose):
    box_size = filterSize * filterSize
    for i in range(len(grad)):
        tmp = torch.take(grad[i], index[i])
        norm_tmp = torch.norm(tmp, dim=-1)
        norm_UD = torch.argsort(norm_tmp,descending=True)
        norm_len = len(norm_tmp)
        choose_ch = [int(norm_len*choose[0]),int(norm_len*choose[1])]
        choose_index = norm_UD[choose_ch[0]:choose_ch[1]]
        norm_0 = torch.zeros_like(norm_tmp).detach().to(device)
        norm_0[choose_index] = 1
        norm_tmp_k = norm_0
        tmp_bi = torch.as_tensor(norm_tmp_k.repeat_interleave(box_size)) * 1.0
        grad[i] = grad[i].put_(index[i], tmp_bi)
    return grad


# Training
print(
    'Label: {} \t Model: {} \t Dataset: {} \t Saving instances: {}'.format(args.target, args.model_type,
                                                                           args.train_dir, epochs))
if TK == True:
    now = 'TK-{}_TG-{}_eps-{}_S-{}_Q-{}_K-{}-box-{}/'.format(args.model_type, args.target, eps, lam_1, lam_2, tk,
                                                             filterSize)
else:
    now = 'CH-{}_TG-{}_eps-{}_S-{}_Q-{}_CH-{}_{}-box-{}/'.format(args.model_type, args.target, eps, lam_1, lam_2,
                                                                 choose[0],choose[1], filterSize)
now_pic = now + 'pictures/'
if not os.path.exists(now):
    os.mkdir(os.path.join(now))
    os.mkdir(os.path.join(now_pic))

out_csv = pd.DataFrame([])
FR_white_box = []
tra_loss, norm_0, norm_1, norm_2, test = [], [], [], [], []
iterp = 2000 // args.batch_size
i_len = train_size // (iterp * args.batch_size)
out_csv['id'] = [i for i in range(i_len * (epochs+1))]

for epoch in range(epochs):
    FR_wb, FR_wb_epoch = 0, 0
    for i, (img, gt) in enumerate(train_loader):

        img = img.to(device)
        gt = gt.to(device)

        if args.target == -1:
            img_in = normalize(img.clone().detach())
            out = model(img_in)
            label = out.argmax(dim=-1).detach()
            out_wb = label.clone().detach()
            out.backward(torch.ones_like(out))
        else:
            out = torch.LongTensor(img.size(0))
            out.fill_(args.target)
            label = out.to(device)

            out_tmp = model(normalize(img.clone().detach()))
            out_tmp.backward(torch.ones_like(out_tmp))
            out_wb = label.clone().detach()

        netG.train()
        optimG.zero_grad()

        grad = back_grad.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
        grad_fea = (grad * back_fea).sum(dim=1)
        resize = transforms.Resize((img_size, img_size), antialias=True)
        G_F = resize(grad_fea).reshape(len(img), 1, img_size, img_size)

        if TK == True:
            grad_box = grad_topk(G_F, index, filterSize, tk)
        else:
            grad_box = grad_choose(G_F,index, filterSize, choose)

        adv, adv_inf, adv_0, adv_00, grad_img = netG(img, grad_box)
        adv_img = adv.clone().detach()
        adv_test = adv.clone().detach()

        adv_out = model(normalize(adv))
        adv_out_to_wb = adv_out.clone().detach()


        if args.target == -1:
            FR_wb_tmp = torch.sum(adv_out_to_wb.argmax(dim=-1) != out_wb).item()
            # Untargeted Attack
            loss_adv = criterion(adv_out, label)
        else:
            FR_wb_tmp = torch.sum(adv_out_to_wb.argmax(dim=-1) == out_wb).item()
            # Targeted Attack
            loss_adv = criterion(adv_out, label, tar=True)

        FR_wb += FR_wb_tmp
        FR_wb_epoch += FR_wb_tmp

        loss_spa = torch.norm(adv_0, 1)
        bi_adv_00 = torch.where(adv_00 < 0.5, torch.zeros_like(adv_00), torch.ones_like(adv_00)*grad_box)
        loss_qua = torch.sum((bi_adv_00 - adv_00) ** 2)
        loss = loss_adv + lam_1 * loss_spa + lam_2 * loss_qua

        loss.backward()
        optimG.step()


        adv_loss = loss_adv
        spa1 = lam_1 * loss_spa
        spa2 = lam_2 * loss_qua

        if i % iterp == 0:
            FR = FR_wb / (iterp * args.batch_size)
            FR_wb = 0
            adv_0_img = torch.where(adv_0 < 0.5, torch.zeros_like(adv_0), torch.ones_like(adv_0)).clone().detach()
            l0 = (torch.norm(adv_0_img.clone().detach(), 0) / args.batch_size).item()
            l1 = (torch.norm(adv_0_img.clone().detach() * adv_inf.clone().detach(), 1) / args.batch_size).item()
            l2 = (torch.norm(adv_0_img.clone().detach() * adv_inf.clone().detach(), 2) / args.batch_size).item()
            linf = (torch.norm(adv_0_img.clone().detach() * adv_inf.clone().detach(), p=np.inf)).item()
            tra_loss.append(loss.item())
            FR_white_box.append(FR)
            norm_0.append(l0)
            norm_1.append(l1)
            norm_2.append(l2)
            print('\n', '#' * 20)
            print('l0:', l0, 'l1:', l1, 'l2:', l2, 'linf:', linf, '\n',
                  'loss: %.4f'%loss.item(),'adv: %.4f'%adv_loss.item(),'spa1: %.4f'%spa1.item(),'spa2:%.4f'%spa2.item(), '\n',
                  args.model_type, ':', FR)

        if epochs < 21:
            try:
                out_csv['tra_loss'] = pd.Series(tra_loss)
                out_csv['norm_0'] = pd.Series(norm_0)
                out_csv['norm_1'] = pd.Series(norm_1)
                out_csv['norm_2'] = pd.Series(norm_2)
                out_csv[args.model_type] = pd.Series(FR_white_box)
                loss_csv = now + "model-{}_eps-{}_lr-{}_S-{}_Q-{}.csv".format(args.model_type, eps, args.lr, lam_1, lam_2)
                out_csv.to_csv(loss_csv)
            except:
                pass

            if i in [200, 1000, 10000, 20000]:
                vutils.save_image(vutils.make_grid(adv_img, normalize=True, scale_each=True),
                                  now_pic + 'ep{}_adv{}.png'.format(epoch, i))
                vutils.save_image(vutils.make_grid(grad_img, normalize=True, scale_each=True),
                                  now_pic + 'ep{}_grad_img{}.png'.format(epoch, i))
                vutils.save_image(vutils.make_grid(adv_img - img, normalize=True, scale_each=True),
                                  now_pic + 'ep{}_noise{}.png'.format(epoch, i))
                vutils.save_image(vutils.make_grid(img, normalize=True, scale_each=True),
                                  now_pic + 'ep{}_org{}.png'.format(epoch, i))
    FR_wb_ep_mean = FR_wb_epoch / train_size
    print('running:{} | FR-{}:{}\n'.format(epoch, args.model_type, FR_wb_ep_mean))
    start, end = int(epoch) * i_len, int(epoch + 1) * i_len
    N0 = np.mean(norm_0[start:end])
    N1 = np.mean(norm_1[start:end])
    try:
        print('loss:{}--L0:{}--L1:{}--L2:{}\n'.format(tra_loss[-1], N0, N1, np.mean(norm_2[start:end])))
    except:
        pass

    save_path = now + 'GN_{}_{}_{}.pth'.format(args.target, args.model_type, epoch)
    torch.save(netG.state_dict(), os.path.join(save_path))

out_csv['tra_loss'] = pd.Series(tra_loss)
out_csv['norm_0'] = pd.Series(norm_0)
out_csv['norm_1'] = pd.Series(norm_1)
out_csv['norm_2'] = pd.Series(norm_2)
out_csv[args.model_type] = pd.Series(FR_white_box)

loss_csv = now + "model-{}_eps-{}_lr-{}_S-{}_Q-{}.csv".format(args.model_type, eps, args.lr, lam_1, lam_2)
out_csv.to_csv(loss_csv)
print("Training completed...")
