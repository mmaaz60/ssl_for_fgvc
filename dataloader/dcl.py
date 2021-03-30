from dataset.dcl import DCL as Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.util import get_object_from_path
import torch


def collate_train(batch):
    imgs = []
    label = []
    label_swap = []
    law_swap = []
    img_name = []
    for sample in batch:
        imgs.append(sample[0])
        imgs.append(sample[1])
        label.append(sample[2])
        label.append(sample[2])
        if sample[3] == -1:
            label_swap.append(1)
            label_swap.append(0)
        else:
            label_swap.append(sample[2])
            label_swap.append(sample[3])
        law_swap.append(sample[4])
        law_swap.append(sample[5])
        img_name.append(sample[-1])
    return torch.stack(imgs, 0), label, label_swap, law_swap, img_name


def collate_test(batch):
    imgs = []
    label = []
    img_name = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])
        img_name.append(sample[-1])
    return torch.stack(imgs, 0), label, img_name


class DCL:
    def __init__(self, config):
        """
        The function parse the configuration parameters and load the CUB_200_2011 dataset
        :param config: YML configuration object
        """
        self.config = config
        self.train_transform = None  # Train transform
        self.test_transform = None  # Test transform
        self.common_transform = None  # Common transforms
        self.jigsaw_transform = None  # Jigsaw transformation
        self.final_transform_train = None  # Final transformation for training
        self.final_transform_test = None  # Final transformation for testing
        self.train_dataset = None  # Train dataset
        self.test_dataset = None  # Test Dataset
        self.get_transforms()  # Parse the train and test transforms from configuration file
        self.load_dataset()  # Load the train and test datasets

    def get_transforms(self):
        common_transforms = self.config.cfg["dataloader"]["transforms"]["common"]
        jigsaw_transforms = self.config.cfg["dataloader"]["transforms"]["jigsaw"]
        final_train_transforms = self.config.cfg["dataloader"]["transforms"]["final_train"]
        final_test_transforms = self.config.cfg["dataloader"]["transforms"]["final_test"]
        self.common_transform = transforms.Compose(
            [
                get_object_from_path(common_transforms[i]['path'])(**common_transforms[i]['param'])
                if 'param' in common_transforms[i].keys()
                else get_object_from_path(common_transforms[i]['path'])() for i in common_transforms.keys()
            ]
        )
        self.jigsaw_transform = transforms.Compose(
            [
                get_object_from_path(jigsaw_transforms[i]['path'])(**jigsaw_transforms[i]['param'])
                if 'param' in jigsaw_transforms[i].keys()
                else get_object_from_path(jigsaw_transforms[i]['path'])() for i in jigsaw_transforms.keys()
            ]
        )
        self.final_transform_train = transforms.Compose(
            [
                get_object_from_path(final_train_transforms[i]['path'])(**final_train_transforms[i]['param'])
                if 'param' in final_train_transforms[i].keys()
                else get_object_from_path(final_train_transforms[i]['path'])() for i in final_train_transforms.keys()
            ]
        )
        self.final_transform_test = transforms.Compose(
            [
                get_object_from_path(final_test_transforms[i]['path'])(**final_test_transforms[i]['param'])
                if 'param' in final_test_transforms[i].keys()
                else get_object_from_path(final_test_transforms[i]['path'])() for i in final_test_transforms.keys()
            ]
        )

    def load_dataset(self):
        """
        Load the train and test datasets
        """
        # Parse configuration
        data_root_directory = self.config.cfg["dataloader"]["root_directory_path"]  # Dataset root directory path
        download = self.config.cfg["dataloader"]["download"]  # Either to download the dataset or not
        train_data_fraction = self.config.cfg["dataloader"]["train_data_fraction"]  # Fraction of dataset for training
        test_data_fraction = self.config.cfg["dataloader"]["test_data_fraction"]  # Fraction of dataset for testing
        # Load the train dataset
        self.train_dataset = Dataset(root=data_root_directory, train=True, download=download,
                                     common_transform=self.common_transform, jigsaw_transform=self.jigsaw_transform,
                                     final_transform=self.final_transform_train,
                                     train_data_fraction=train_data_fraction)
        # Load the test dataset
        self.test_dataset = Dataset(root=data_root_directory, train=False, download=download,
                                    final_transform=self.final_transform_test, test_data_fraction=test_data_fraction)

    def get_dataloader(self):
        """
        Create and return train and test dataloaders
        :return: train_dataloader, test_dataloader
        """
        # Parse configuration
        batch_size = self.config.cfg["dataloader"]["batch_size"]  # Batch size for the dataloader
        shuffle = self.config.cfg["dataloader"]["shuffle"]  # Either to shuffle dataset or not
        num_workers = self.config.cfg["dataloader"]["num_workers"]  # Number of workers to load the dataset
        # Create the train dataloader
        train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, collate_fn=collate_train,
                                      shuffle=shuffle, num_workers=num_workers)
        # Create the test dataloader
        test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=batch_size, collate_fn=collate_test,
                                     shuffle=shuffle, num_workers=num_workers)
        # Return train and test dataloader
        return train_dataloader, test_dataloader
