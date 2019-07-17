from data_loader import get_data_loader
from torchvision.utils import save_image
import torchvision.transforms as T
import os
import argparse

_TFS = T.Compose([
        T.Resize(28),
        T.ToTensor(),
        T.Normalize((.5,), (.5,))
    ])


def main(opt):
    svhn_corpus = get_data_loader('svhn', opt.root, tfs=_TFS, batch_size=1, train_flag=True)
    for idx, (img, label) in enumerate(svhn_corpus):
        if not os.path.exists('data/svhn_png/training/%s/' % str(label.item())):
            os.mkdir('data/svhn_png/training/%s/' % str(label.item()))
        save_image(img, 'data/svhn_png/training/%s/%s.png' % (str(label.item()), idx))


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