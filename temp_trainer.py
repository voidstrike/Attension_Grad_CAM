from model.LeNet import LeNet
import torch
import argparse
from torch import nn
from torch.autograd import Variable
from data_loader import get_data_loader
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

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


class Trainer(object):
    def __init__(self, model, dl, optimizer, criterion, model_path, display=10):
        self.gpu = torch.cuda.is_available()
        self.model = model
        self.model_path = model_path
        self.dl = dl
        self.optim = optimizer
        self.criterion = criterion
        self.display_epoch = display

        if self.gpu:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

    def train(self, epochs):
        self.model.train()

        for epoch in range(epochs):
            for _, (f, label) in enumerate(self.dl):
                f = Variable(f)
                label = Variable(label)

                if self.gpu:
                    f = f.cuda()
                    label = label.cuda()

                res = self.model(f)
                loss = self.criterion(res['logit'], label)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            if epoch % self.display_epoch == 0:
                print('Epoch: %s' % (str(epoch + 1)))
                self.test()

    def test(self):
        self.model.eval()
        clf_acc, instance_count = .0, self.dl.dataset.__len__()

        for _, (f, label) in enumerate(self.dl):
            f = Variable(f)
            label = Variable(label)

            if self.gpu:
                f = f.cuda()
                label = label.cuda()

            res = self.model(f)
            _, pred_label = res['logit'].max(1)
            num_correct = (label == pred_label).sum().item()
            clf_acc += num_correct

        clf_acc /= instance_count
        print("Current Accuracy is: %.6f" % clf_acc)
        self.model.train()

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def grad_cam(self, img, id=0):
        self.model.eval()
        # print(img)
        img, img_path = Variable(img[0]).unsqueeze(0), img[2]

        if self.gpu:
            img = img.cuda()

        res = self.model(img)
        pred = res['logit']
        tgt = pred.argmax(dim=1)

        # print(pred.shape)
        # print(tgt)
        # print(pred[:, tgt])
        pred[:, tgt.data].backward()
        gradients = self.model.get_activations_gradient()

        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        activations = res['c2'].detach()

        for i in range(16):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()

        heatmap = np.maximum(heatmap.cpu(), 0)

        heatmap /= torch.max(heatmap)

        # plt.figure(figsize=(10, 10))
        # ax1 = plt.subplot(121)
        # ax1.show(plt.matshow(heatmap.squeeze()))

        heatmap = heatmap.numpy()
        img = cv2.imread(img_path)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + img
        cv2.imwrite('temp_res/map_{}.jpg'.format(id), superimposed_img)

        # plt.matshow(heatmap.squeeze())
        # plt.show()


def main(opt):
    global _TFS_DIC
    root_path = opt.root
    data_name = opt.src

    data_loader = get_data_loader(data_name, root_path, tfs=_TFS_DIC[data_name])
    data_loader_v2 = get_data_loader('svhn_transfer', root_path, tfs= _TFS_DIC[opt.tgt])

    network = LeNet()
    if opt.epoch == 0:
        network.load_state_dict(torch.load('model_info/nn.pt'))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=opt.lr, weight_decay=opt.reg, betas=(opt.beta1, opt.beta2))
    trainer = Trainer(network, data_loader, optimizer, criterion, 'model_info/nn.pt')

    trainer.train(opt.epoch)
    trainer.save_model()

    # trainer.test()
    # trainer.dl = data_loader_v2
    for i in range(100):
        idx = random.randint(0, data_loader.dataset.__len__() - 1)
        trainer.grad_cam(data_loader_v2.dataset[idx], idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, default='data/')
    parser.add_argument('--src', type=str, default='mnist')
    parser.add_argument('--tgt', type=str, default='svhn')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--reg', type=float, default=2.5e-5)
    parser.add_argument('--beta1', type=float, default=.5)
    parser.add_argument('--beta2', type=float, default=.99)
    parser.add_argument('--epoch', type=int, default=100)

    parser.add_argument('--clf_loss', type=float, default=10.0)
    parser.add_argument('--src_rec_loss', type=float, default=1.0)
    parser.add_argument('--tgt_rec_loss', type=float, default=1.0)
    parser.add_argument('--df_loss', type=float, default=10.0)

    parser.add_argument('--display', type=int, default=10)

    parser.add_argument('--log', type=str, default='log/')
    parser.add_argument('--feature_name', type=str, default='cnn.pt')
    parser.add_argument('--clf_name', type=str, default='clf.pt')

    opt = parser.parse_args()
    main(opt)
