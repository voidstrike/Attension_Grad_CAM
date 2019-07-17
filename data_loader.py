import os

from torchvision import datasets
from torch.utils.data import DataLoader
from usps import USPS


_DS_NAME = ['mnist', 'usps', 'svhn', 'office', 'cifar']
_DSS_NAME = ['amazon', 'webcam', 'dslr']


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def get_data_loader(ds_name, root_path, batch_size=32, tfs=None, train_flag=True, dss_name=None):
    global _DS_NAME, _DSS_NAME

    if ds_name not in _DS_NAME:
        raise Exception('Unsupported data set')

    if ds_name == 'mnist':
        # ds = datasets.MNIST(os.path.join(root_path, ds_name), train=train_flag, transform=tfs, download=True)
        if train_flag:
            # ds = datasets.ImageFolder(os.path.join(root_path, ds_name + "_png/training/"), transform=tfs)
            ds = ImageFolderWithPaths(os.path.join(root_path, ds_name + "_png/training/"), transform=tfs)
        else:
            # ds = datasets.ImageFolder(os.path.join(root_path, ds_name + "_png/testing/"), transform=tfs)
            ds = ImageFolderWithPaths(os.path.join(root_path, ds_name + "_png/testing/"), transform=tfs)
    elif ds_name == 'svhn':
        # ds = datasets.SVHN(os.path.join(root_path, ds_name), split='train' if train_flag else 'test',
        #                    transform=tfs, download=True)
        if train_flag:
            # ds = datasets.ImageFolder(os.path.join(root_path, ds_name + "_png/training/"), transform=tfs)
            ds = ImageFolderWithPaths(os.path.join(root_path, ds_name + "_png/training/"), transform=tfs)
        else:
            # ds = datasets.ImageFolder(os.path.join(root_path, ds_name + "_png/testing/"), transform=tfs)
            ds = ImageFolderWithPaths(os.path.join(root_path, ds_name + "_png/testing/"), transform=tfs)
    elif ds_name == 'usps':
        ds = USPS(root_path, train=train_flag, transform=tfs, download=True)
    elif ds_name == 'cifar':
        ds = datasets.CIFAR10(os.path.join(root_path, ds_name), train=train_flag, transform=tfs, download=True)
    elif ds_name == 'office':
        # Modification may be required, rewrite it via OFFICE class
        assert dss_name is not None and dss_name in _DSS_NAME

        tmp_path = os.path.join(root_path, ds_name)
        tmp_path = os.path.join(tmp_path, 'train' if train_flag else 'test')
        ds = datasets.ImageFolder(os.path.join(tmp_path, dss_name), transform=tfs)

    return DataLoader(ds, batch_size=batch_size, shuffle=True)
