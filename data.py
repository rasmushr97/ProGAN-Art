import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torchvision.datasets as datasets
from PIL import Image
import h5py

def create_MNIST_dataloader(img_size, batch_size):
    transform = transforms.Compose([
        transforms.Pad((2, 2)),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=8, pin_memory=True)
    return dataloader

def create_imagefolder_dataloader(image_folder, img_size, batch_size):
    dataset = datasets.ImageFolder(root=image_folder,
                               transform=transforms.Compose([
                                   transforms.Resize(img_size),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=4, 
                                             pin_memory=True)
    return dataloader

def create_h5_dataloader(h5_file, img_size, batch_size):
    transform = transforms.Compose([
                                   transforms.Resize(img_size),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ])

    dataset = HDF5Dataset(h5_file, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=0)
    return dataloader


class HDF5Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super(HDF5Dataset, self).__init__()

        self.file = h5py.File(root_dir, mode='r')
        self.n_images, self.rows, self.cols, self.channels = self.file['images'].shape
        self.transform = transform

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        image = self.file['images'][idx, :, :]
        image = Image.fromarray(image.astype('uint8'), 'RGB')

        if self.transform:
            image = self.transform(image)

        return image

    