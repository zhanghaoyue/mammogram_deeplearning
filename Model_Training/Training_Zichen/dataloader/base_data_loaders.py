import warnings

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from skimage import io
from skimage import transform
from torch.utils.data import Dataset

basic_transform_img = transforms.Compose([transforms.Resize((512, 512)),
                                          transforms.ToTensor()])

basic_transform_mask = transforms.Compose([transforms.ToTensor()])


def basic_transform(image, mask):
    image = basic_transform_img(image)
    mask_resized = np.stack([transform.resize(sub_mask,
                                              (512, 512),
                                              mode='edge',
                                              anti_aliasing=False,
                                              anti_aliasing_sigma=None,
                                              preserve_range=True,
                                              order=0) for sub_mask in mask], 2)
    mask_binary = ((mask_resized > 0) * 255).astype('uint8')

    mask = basic_transform_mask(mask_binary)

    return image, mask


class INBreastDataset(Dataset):

    def __init__(self, image_list, mask_list, phase, transform=basic_transform):
        self.image_list = image_list
        self.mask_list = mask_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx])
        mask = [io.imread(sub_mask, as_gray=True) for sub_mask in self.mask_list[idx]]
        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            warnings.warn('Transformation function is needed!')

        return image, mask


class UCLADataset(Dataset):

    def __init__(self, image_list, label_list, phase, transform=basic_transform_img):
        self.image_list = image_list
        self.label_list = label_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return self.image_list.shape[0]

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        mass_region_attention_path = image_path.replace('preprocess', 'mass_region_attention')

        image = Image.open(image_path)
        mass_region_attention = Image.open(mass_region_attention_path)
        label = self.label_list[idx]

        if self.transform:
            image = self.transform(image)
            mass_region_attention = self.transform(mass_region_attention)

        return image, mass_region_attention, label
