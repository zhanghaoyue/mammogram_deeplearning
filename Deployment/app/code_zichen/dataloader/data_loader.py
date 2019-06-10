import glob
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .base_data_loaders import base_data_loader


def get_mammography_dataset(config):
    config_data_loader = config['data_loader']['args']
    image_root = config_data_loader['image_root']
    label_path = config_data_loader['label_path']

    image_list = np.array(glob.glob(image_root + '*.png'))
    label_table = pd.read_csv(label_path).values[:, -3:]

    name_list = [path.split('/')[-1].split('.')[0] for path in image_list]
    index_list = np.array([np.where(label_table[:, 0] == name)[0][0] for name in name_list])
    label_list = np.array([np.sum(label_table[index][1:]) for index in index_list])

    pos_index = np.where(label_list == 1)[0].tolist()
    neg_index = np.where(label_list == 0)[0]
    neg_index_balanced = neg_index[:len(pos_index)].tolist()

    x = image_list[pos_index + neg_index_balanced]
    y = label_list[pos_index + neg_index_balanced]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)

    print('train set :%d, val set:%d' % (len(x_train), len(x_test)))
    train_set = base_data_loader(config, x_train, y_train, 'train')
    val_set = base_data_loader(config, x_test, y_test, 'val')

    return train_set, val_set


if __name__ == '__main__':
    config = json.load(open('../config/config.json'))
    train_set, val_set = get_mammography_dataset(config)
    a = iter(train_set)
    data = next(a)
    inputs, labels = data['image'], data['label']
    print(len(val_set))
