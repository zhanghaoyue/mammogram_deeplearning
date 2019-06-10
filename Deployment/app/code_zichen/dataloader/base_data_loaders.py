from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

image_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Resize([224, 224]),
    # transforms.RandomCrop([224, 224]),
    transforms.ToTensor()])


class MyDataset(Dataset):

    def __init__(self, image_list, label_list, transform=None):
        self.image_list = image_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return self.image_list.shape[0]

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx])
        label = self.label_list[idx]
        sample = {'image': image, 'label': label}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample


def base_data_loader(config, image_list, label_list, phase):
    data_loader_config = config['data_loader']['args']
    batch_size = data_loader_config['train_batch_size'] if phase == 'train' else data_loader_config['test_batch_size']

    dataset = MyDataset(image_list, label_list, transform=image_transform)
    data_loader = DataLoader(dataset, batch_size, shuffle=(phase == 'train'),
                             num_workers=8, drop_last=False)

    return data_loader
