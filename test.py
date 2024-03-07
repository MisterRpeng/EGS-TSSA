import argparse
import os
import numpy as np
import torchvision
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from test_generators import GeneratorResnet


parser = argparse.ArgumentParser(description='testing EGS_TSSA for generating sparse adversarial examples')
parser.add_argument('--test_dir', default='Dataset/val_data/', help='path to imagenet testing set')
parser.add_argument('--model_type', type=str, default='res50', help='Model against GAN is tested: incv3, res50')
parser.add_argument('--model_t', type=str, default='vgg16', help='Model')
parser.add_argument('--eps', type=int, default=10, help='Perturbation Budget')
parser.add_argument('--target', type=int, default=-1, help='-1 if untargeted')
parser.add_argument('--batch_size', type=int, default=1, help='Number of testig samples/batch')
parser.add_argument('--checkpoint', type=str, default='weights/soft_eps255_res50_tk0.873.pth', help='path to checkpoint')
parser.add_argument('--tk', type=float, default=0.873, help='path to checkpoint')

if __name__ == '__main__':
    args = parser.parse_known_args()[0]
    eps = args.eps
    print(args)

    tk = args.tk
    choose = [0., 0.6]

    print('eps:', eps)

    # GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    back_fea = torch.tensor([]).to(device)
    back_grad = torch.tensor([]).to(device)


    # Getting the gradient
    def backward_hook(module, grad_in, grad_out):
        global back_grad
        back_grad = grad_out[0].clone().detach()


    # Get feature layer
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

    if args.model_t == 'dense161':
        model_t = torchvision.models.densenet161(pretrained=True)
    elif args.model_t == 'vgg16':
        model_t = torchvision.models.vgg16(pretrained=True)
    elif args.model_t == 'incv3':
        model_t = torchvision.models.inception_v3(pretrained=True)
    elif args.model_t == 'res50':
        model_t = torchvision.models.resnet50(pretrained=True)

    model_t = model_t.to(device)
    model_t.eval()

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
    netG.load_state_dict(torch.load(args.checkpoint, map_location='cuda:0'))
    netG.to(device)
    netG.eval()

    # Data
    data_transform = transforms.Compose([
        transforms.Resize(scale_size, antialias=True),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])


    def trans_incep(x):
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        return x


    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]


    def normalize(t):
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]

        return t


    test_set = datasets.ImageFolder(args.test_dir, data_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                               pin_memory=True)
    test_size = len(test_set)
    print('test data size:', test_size)

    # Get the most important area
    def grad_topk(grad, index, filterSize, Tk=tk):
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

    # Get the zone area of interest
    def grad_choose(grad, index, filterSize, choose):
        box_size = filterSize * filterSize
        for i in range(len(grad)):
            tmp = torch.take(grad[i], index[i])
            norm_tmp = torch.norm(tmp, dim=-1)
            norm_UD = torch.argsort(norm_tmp, descending=True)
            norm_len = len(norm_tmp)
            choose_ch = [int(norm_len * choose[0]), int(norm_len * choose[1])]
            choose_index = norm_UD[choose_ch[0]:choose_ch[1]]
            norm_0 = torch.zeros_like(norm_tmp).detach().to(device)
            norm_0[choose_index] = 1
            norm_tmp_k = norm_0
            tmp_bi = torch.as_tensor(norm_tmp_k.repeat_interleave(box_size)) * 1.0
            grad[i] = grad[i].put_(index[i], tmp_bi)
        return grad


    now = '{}TO{}_eps-{}-K-{}/'.format(args.model_type, args.model_t, eps, tk)
    now_pic = now + 'pictures/'
    if not os.path.exists(now):
        os.mkdir(os.path.join(now))
        os.mkdir(os.path.join(now_pic))
    l0, l1, l2, linf = 0, 0, 0, 0
    FR_bb_epoch, FR_wb_epoch = 0, 0
    for i, (img, gt) in enumerate(test_loader):
        img = img.to(device)
        gt = gt.to(device)

        if 'inc' in args.model_type or 'xcep' in args.model_type:
            out = model(normalize(trans_incep(img.clone().detach())))

        else:
            out = model(normalize(img.clone().detach()))
        label = out.argmax(dim=-1).clone().detach()
        out_wb = label.clone().detach()
        out.backward(torch.ones_like(out))

        if 'inc' in args.model_t or 'xcep' in args.model_t:
            out_bb = model_t(normalize(trans_incep(img.clone().detach())))
        else:
            out_bb = model_t(normalize(img.clone().detach()))

        # Getting a structured mask
        grad = back_grad.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
        grad_fea = (grad * back_fea).sum(dim=1)
        resize = transforms.Resize((img_size, img_size), antialias=True)
        G_F = resize(grad_fea).reshape(len(img), 1, img_size, img_size)
        # grad_box = grad_choose(G_F, index, filterSize, choose)
        grad_box = grad_topk(G_F, index, filterSize, tk)

        adv, adv_inf, adv_0, adv_00, grad_img = netG(img, grad_box)
        adv_img = adv.clone().detach()
        adv_test = adv.clone().detach()

        if 'inc' in args.model_type or 'xcep' in args.model_type:
            adv_out = model(normalize(trans_incep(adv.clone().detach())))

        else:
            adv_out = model(normalize(adv.clone().detach()))

        adv_out_to_wb = adv_out.clone().detach()
        if 'inc' in args.model_t or 'xcep' in args.model_t:
            adv_out_to_bb = model_t(normalize(trans_incep(adv_test.clone().detach())))
        else:
            adv_out_to_bb = model_t(normalize(adv_test.clone().detach()))

        if args.target == -1:
            FR_wb_tmp = torch.sum(adv_out_to_wb.argmax(dim=-1) != out_wb).item()
            FR_bb_tmp = torch.sum(adv_out_to_bb.argmax(dim=-1) != out_bb.argmax(dim=-1)).item()

        else:
            FR_wb_tmp = torch.sum(adv_out_to_wb.argmax(dim=-1) == out_wb).item()
            FR_bb_tmp = torch.sum(adv_out_to_bb.argmax(dim=-1) == out_bb.argmax(dim=-1)).item()

        FR_wb_epoch += FR_wb_tmp
        FR_bb_epoch += FR_bb_tmp

        l0 += torch.norm(adv_0.clone().detach(), 0).item()
        l1 += torch.norm(adv_0.clone().detach() * adv_inf.clone().detach(), 1).item()
        l2 += torch.norm(adv_0.clone().detach() * adv_inf.clone().detach(), 2).item()
        linf = (torch.norm(adv_0.clone().detach() * adv_inf.clone().detach(), p=np.inf)).item()

        if i in [201, 1001, 2001, 3001, 4001]:
            vutils.save_image(vutils.make_grid(adv_img, normalize=True, scale_each=True),
                              now_pic + 'adv{}.png'.format(i))
            vutils.save_image(vutils.make_grid(grad_img, normalize=True, scale_each=True),
                              now_pic + 'grad_img{}.png'.format(i))
            vutils.save_image(vutils.make_grid(adv_img - img, normalize=True, scale_each=True),
                              now_pic + 'noise{}.png'.format(i))
            vutils.save_image(vutils.make_grid(img, normalize=True, scale_each=True),
                              now_pic + 'org{}.png'.format(i))

    FR_wb_ep_mean = FR_wb_epoch / test_size
    FR_bb_ep_mean = FR_bb_epoch / test_size
    print('FR-{}:{} | FR-{}:{}\n'.format(args.model_type, FR_wb_ep_mean, args.model_t,
                                         FR_bb_ep_mean))

    try:
        print('L0:{}--L1:{:.4f}--L2:{:.4f}--Linf:{:.4f}\n'.format(int(l0 / test_size), l1 / test_size, l2 / test_size,
                                                                  linf))
    except:
        pass