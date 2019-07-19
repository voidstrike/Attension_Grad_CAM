from model.LeAE import *
from data_loader import get_data_loader

import torch
import torchvision.transforms as T
import argparse
import os
import cv2
import numpy as np
import random

from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision.utils import save_image


_TFS_DIC = {
    'mnist': T.Compose([
        T.Grayscale(),
        T.ToTensor(),
        T.Normalize((.5,), (.5,))
    ]),
    'usps': T.Compose([
        T.ToTensor(),
        T.Normalize((.5,), (.5,))
    ]),
    'svhn': T.Compose([
        T.Grayscale(),
        T.ToTensor(),
        T.Normalize((.5,), (.5,))
    ]),
    'cifar': T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
}


def convert2img(tran_net):
    tran_net.eval()
    svhn_corpus = get_data_loader('svhn', opt.root, tfs=_TFS_DIC['svhn'], batch_size=1, train_flag=True)
    for idx, (img, label, _) in enumerate(svhn_corpus):
        if not os.path.exists('data/svhn_gray/training/%s/' % str(label.item())):
            os.mkdir('data/svhn_gray/training/%s/' % str(label.item()))

        img = Variable(img)
        if torch.cuda.is_available():
            img = img.cuda()

        # out = tran_net(img)['rec']
        save_image(img.data, 'data/svhn_gray/training/%s/%s.png' % (str(label.item()), idx))
    tran_net.train()


# function that create a heat map using Grad-CAM
def dis_heatmap(dis_net, img, idx=0):
    dis_net.eval()

    img, img_path = Variable(img[0]).unsqueeze(0), img[2]

    if torch.cuda.is_available():
        img = img.cuda()

    res = dis_net(img)
    # pred = res['logit']
    pred = res
    tgt = pred.argmax(dim=1)

    pred[:, tgt.data].backward()
    gradients = dis_net.get_activations_gradient()

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    activations = res.detach()

    for i in range(2):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap.cpu(), 0)
    heatmap /= torch.max(heatmap)

    heatmap = heatmap.numpy()
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    cv2.imwrite('temp_res/map_{}.jpg'.format(idx), superimposed_img)

    # plt.matshow(heatmap.squeeze())
    # plt.show()


def main(opt):
    src_tgt, tgt_src = LeAE(), LeAE()
    src_dis, tgt_dis = LeDiscriminator(), LeDiscriminator()

    if opt.epoch == 0:
        src_tgt.load_state_dict(torch.load('model_info/src2tgt.pt'))
        tgt_src.load_state_dict(torch.load('model_info/tgt2src.pt'))
        src_dis.load_state_dict(torch.load('model_info/src_dis.pt'))
        tgt_dis.load_state_dict(torch.load('model_info/tgt_dis.pt'))

    src_dl = get_data_loader(opt.src, opt.root, tfs=_TFS_DIC[opt.src])
    tgt_dl = get_data_loader(opt.tgt, opt.root, tfs=_TFS_DIC[opt.tgt])

    # criterion_GAN = nn.MSELoss()
    criterion_gan_v2 = nn.CrossEntropyLoss()
    criterion_cycle = nn.L1Loss()
    criterion_rec = nn.L1Loss()
    criterion_dis = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        src_tgt, tgt_src = src_tgt.cuda(), tgt_src.cuda()
        src_dis, tgt_dis = src_dis.cuda(), tgt_dis.cuda()
        # criterion_GAN = criterion_GAN.cuda()
        criterion_cycle = criterion_cycle.cuda()
        criterion_rec = criterion_rec.cuda()

    optimizer_G = optim.Adam(list(src_tgt.parameters()) + list(tgt_src.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_Ds = optim.Adam(src_dis.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_Dt = optim.Adam(tgt_dis.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Standard Cyc-GAN training process
    for epoch in range(opt.epoch):
        for idx, ((f_s, _, _), (f_t, _, _)) in enumerate(zip(src_dl, tgt_dl)):
            f_s, f_t = Variable(f_s), Variable(f_t)
            src_real = Variable(torch.ones(f_s.size(0), dtype=torch.int64))
            src_fake = Variable(torch.zeros(f_s.size(0), dtype=torch.int64))
            tgt_real = Variable(torch.ones(f_t.size(0), dtype=torch.int64))
            tgt_fake = Variable(torch.zeros(f_t.size(0), dtype=torch.int64))

            if torch.cuda.is_available():
                f_s, f_t = f_s.cuda(), f_t.cuda()
                src_real, src_fake = src_real.cuda(), src_fake.cuda()
                tgt_real, tgt_fake = tgt_real.cuda(), tgt_fake.cuda()

            for _ in range(2):
                # Train Generator
                optimizer_G.zero_grad()

                src2tgt_rec_loss = criterion_rec(tgt_src(f_s)['rec'], f_s)
                tgt2src_rec_loss = criterion_rec(src_tgt(f_t)['rec'], f_t)
                rec_loss = src2tgt_rec_loss + tgt2src_rec_loss

                fake_tgt, fake_src = src_tgt(f_s)['rec'], tgt_src(f_t)['rec']
                # src2tgt_gan_loss = criterion_GAN(tgt_dis(fake_tgt)[:, 0].view(-1), src_real)
                # tgt2src_gan_loss = criterion_GAN(src_dis(fake_src)[:, 0].view(-1), tgt_real)
                src2tgt_gan_loss = criterion_gan_v2(tgt_dis(fake_tgt), src_real)
                tgt2src_gan_loss = criterion_gan_v2(src_dis(fake_src), tgt_real)
                gan_loss = src2tgt_gan_loss + tgt2src_gan_loss

                rec_src, rec_tgt = tgt_src(fake_tgt)['rec'], src_tgt(fake_src)['rec']
                src2src_cyc_loss = criterion_cycle(rec_src, f_s)
                tgt2tgt_cyc_loss = criterion_cycle(rec_tgt, f_t)
                cyc_loss = src2src_cyc_loss + tgt2tgt_cyc_loss

                loss_G = gan_loss + opt.cyc_fac * cyc_loss + opt.rec_fac * rec_loss
                loss_G.backward()

                optimizer_G.step()

            # Train SRC-DIS
            optimizer_Ds.zero_grad()
            pred_label = torch.cat((src_dis(f_s), src_dis(tgt_src(f_t.detach())['rec'])), 0)
            real_label = torch.cat((src_real.long(), tgt_fake.long()), 0)
            # dis_loss = criterion_GAN(pred_label, real_label)
            dis_loss = criterion_dis(pred_label, real_label)

            dis_loss.backward()
            optimizer_Ds.step()

            # Train TGT-DIS
            optimizer_Dt.zero_grad()
            pred_label = torch.cat((tgt_dis(f_t), tgt_dis(src_tgt(f_s.detach())['rec'])), 0)
            real_label = torch.cat((tgt_real.long(), src_fake.long()), 0)
            # dis_loss = criterion_GAN(pred_label, real_label)
            dis_loss = criterion_dis(pred_label, real_label)

            dis_loss.backward()
            optimizer_Dt.step()

        if epoch % opt.display == 0:
            # Duplicate Test Code
            src_tgt, tgt_src = src_tgt.eval(), tgt_src.eval()
            src_dis, tgt_dis = src_dis.eval(), tgt_dis.eval()

            total_rec, total_gan, total_cyc = .0, .0, .0
            total_src_dis, total_tgt_dis = .0, .0
            batch_count = .0

            for idx, ((f_s, _, _), (f_t, _, _)) in enumerate(zip(src_dl, tgt_dl)):
                batch_count += 1.0

                f_s, f_t = Variable(f_s), Variable(f_t)
                src_real = Variable(torch.ones(f_s.size(0), dtype=torch.int64))
                src_fake = Variable(torch.zeros(f_s.size(0), dtype=torch.int64))
                tgt_real = Variable(torch.ones(f_t.size(0), dtype=torch.int64))
                tgt_fake = Variable(torch.zeros(f_t.size(0), dtype=torch.int64))

                if torch.cuda.is_available():
                    f_s, f_t = f_s.cuda(), f_t.cuda()
                    src_real, src_fake = src_real.cuda(), src_fake.cuda()
                    tgt_real, tgt_fake = tgt_real.cuda(), tgt_fake.cuda()

                # Train Generator
                src2tgt_rec_loss = criterion_rec(tgt_src(f_s)['rec'], f_s)
                tgt2src_rec_loss = criterion_rec(src_tgt(f_t)['rec'], f_t)
                rec_loss = src2tgt_rec_loss + tgt2src_rec_loss
                total_rec += rec_loss.item()

                fake_tgt, fake_src = src_tgt(f_s)['rec'], tgt_src(f_t)['rec']
                src2tgt_gan_loss = criterion_gan_v2(tgt_dis(fake_tgt), src_real)
                tgt2src_gan_loss = criterion_gan_v2(src_dis(fake_src), tgt_real)
                gan_loss = src2tgt_gan_loss + tgt2src_gan_loss
                total_gan += gan_loss.item()

                rec_src, rec_tgt = tgt_src(fake_tgt)['rec'], src_tgt(fake_src)['rec']
                src2src_cyc_loss = criterion_cycle(rec_src, f_s)
                tgt2tgt_cyc_loss = criterion_cycle(rec_tgt, f_t)
                cyc_loss = src2src_cyc_loss + tgt2tgt_cyc_loss
                total_cyc += cyc_loss.item()

                pred_label = torch.cat((src_dis(f_s), src_dis(tgt_src(f_t.detach())['rec'])), 0)
                real_label = torch.cat((src_real.long(), tgt_fake.long()), 0)
                # dis_loss = criterion_GAN(pred_label, real_label)
                dis_loss = criterion_dis(pred_label, real_label)
                total_src_dis += dis_loss.item()

                pred_label = torch.cat((tgt_dis(f_t), tgt_dis(src_tgt(f_s.detach())['rec'])), 0)
                real_label = torch.cat((tgt_real.long(), src_fake.long()), 0)
                # dis_loss = criterion_GAN(pred_label, real_label)
                dis_loss = criterion_dis(pred_label, real_label)
                total_tgt_dis += dis_loss.item()

            print('Transfer Loss : {:.6f}, GAN Loss : {:.6f}, Cycle Loss : {:.6f}'.
                  format(total_rec / batch_count, total_gan / batch_count, total_cyc / batch_count))
            print("SRC-DIS Loss : {:.6f}".format(total_src_dis / batch_count))
            print("TGT-DIS Loss : {:.6f}".format(total_tgt_dis / batch_count))

            src_tgt, tgt_src = src_tgt.train(), tgt_src.train()
            src_dis, tgt_dis = src_dis.train(), tgt_dis.train()

    # convert2img(tgt_src)
    torch.save(src_tgt.state_dict(), 'model_info/src2tgt.pt')
    torch.save(tgt_src.state_dict(), 'model_info/tgt2src.pt')
    torch.save(src_dis.state_dict(), 'model_info/src_dis.pt')
    torch.save(tgt_dis.state_dict(), 'model_info/tgt_dis.pt')

    for _ in range(100):
        idx = random.randint(0, tgt_dl.dataset.__len__() - 1)
        dis_heatmap(tgt_dis, tgt_dl.dataset[idx], idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, default='data/')
    parser.add_argument('--src', type=str, default='mnist')
    parser.add_argument('--tgt', type=str, default='svhn')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--reg', type=float, default=2.5e-5)
    parser.add_argument('--b1', type=float, default=.5)
    parser.add_argument('--b2', type=float, default=.99)
    parser.add_argument('--epoch', type=int, default=100)

    parser.add_argument('--cyc_fac', type=float, default=10.0)
    parser.add_argument('--rec_fac', type=float, default=5.0)

    parser.add_argument('--display', type=int, default=10)

    parser.add_argument('--log', type=str, default='log/')
    parser.add_argument('--feature_name', type=str, default='cnn.pt')
    parser.add_argument('--clf_name', type=str, default='clf.pt')

    opt = parser.parse_args()
    main(opt)
