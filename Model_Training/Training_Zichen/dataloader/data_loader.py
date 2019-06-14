import glob

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .base_data_loaders import *


def get_INBreast_dataloader(config):
    config_dataloader = config['data_loader']['args']

    image_list = sorted(glob.glob('/home/zcwang/Desktop/local-projects/BE223c/data/INBreast/preprocess/image/*.png'))
    mass_list = [path.replace('image', 'mass') for path in image_list]
    muscle_list = [path.replace('image', 'muscle') for path in image_list]

    mask_list = np.stack([mass_list, muscle_list], 1)
    x_train, x_test, y_train, y_test = train_test_split(image_list, mask_list, test_size=0.2, random_state=0)

    train_set = INBreastDataset(x_train, y_train, 'train')
    test_set = INBreastDataset(x_test, y_test, 'test')

    train_dataloader = DataLoader(train_set, config_dataloader['train_batch_size'], shuffle=True,
                                  num_workers=config_dataloader['num_workers'],
                                  drop_last=config_dataloader['drop_last'])
    test_dataloader = DataLoader(test_set, config_dataloader['test_batch_size'], shuffle=False,
                                 num_workers=config_dataloader['num_workers'],
                                 drop_last=config_dataloader['drop_last'])

    return train_dataloader, test_dataloader


def split_UCLA_dataset():
    image_list = np.load('/home/zcwang/Desktop/local-projects/BE223c/data/tmp/selected_image_path.npy')
    label_list = np.load('/home/zcwang/Desktop/local-projects/BE223c/data/tmp/selected_label.npy')

    pos_index = np.where(label_list == 1)[0].tolist()
    neg_index = np.where(label_list == 0)[0]
    neg_index_balanced = neg_index[:len(pos_index)].tolist()

    x = image_list[pos_index + neg_index_balanced]
    y = label_list[pos_index + neg_index_balanced]

    return x, y


def get_UCLA_dataset(config):
    config_dataloader = config['data_loader']['args']

    x, y = split_UCLA_dataset()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)

    print('length of train, length of val, train_pos, val_pos')
    print(len(y_train), np.sum(y_train), len(y_test), np.sum(y_test))

    train_set = UCLADataset(x_train, y_train, 'train')
    test_set = UCLADataset(x_test, y_test, 'test')

    train_dataloader = DataLoader(train_set, config_dataloader['train_batch_size'], shuffle=True,
                                  num_workers=config_dataloader['num_workers'],
                                  drop_last=config_dataloader['drop_last'])
    test_dataloader = DataLoader(test_set, config_dataloader['test_batch_size'], shuffle=False,
                                 num_workers=config_dataloader['num_workers'],
                                 drop_last=config_dataloader['drop_last'])

    return train_dataloader, test_dataloader


if __name__ == '__main__':
    import torchvision.transforms as transforms
    import numpy as np
    import json
    import matplotlib.pyplot as plt

    config_path = '/home/zcwang/Desktop/local-projects/BE223c/code/config/config.json'
    config = json.load(open(config_path))

    # train_dataloader, test_dataloader = get_INBreast_dataloader(config)
    # for data in iter(train_dataloader):
    #     image, mask = data
    #     break
    #
    # print(np.unique(mask))
    # print(mask.shape, image.shape)
    #
    # plt.figure()
    # plt.imshow(mask[0][0])
    # plt.show()
    #
    # plt.figure()
    # plt.imshow(mask[0][1])
    # plt.show()
    #
    # plt.figure()
    # plt.imshow(image[0][0])
    # plt.show()

    train_dataloader, test_dataloader = get_UCLA_dataset(config)
    for data in iter(train_dataloader):
        image, label = data
        break
    print(label)
    plt.figure()
    plt.imshow(image[0][0])
    plt.show()
