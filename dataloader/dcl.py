from dataset.dcl import DCL as Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.util import get_object_from_path
import torch


def collate_train(batch):
    """
    Collate function for preparing the training batch for DCL training.

    :param batch: The list containing outputs of the __get_item__() function.
    Length of the list is equal to the required batch size.
    :return: The batch containing images and labels for DCL training
    """
    imgs = []  # List to store the images
    target = []  # List to store the targets
    target_jigsaw = []  # List to store the jigsaw targets
    patch_labels = []  # List to store the ground truth patch labels
    # Iterate over each output of the __get_item__() function
    for sample in batch:
        imgs.append(sample[0])  # Append the original image
        imgs.append(sample[1])  # Append the transformed image (after applying RCM)
        # Append the same labels for original and RCM image for Cls. head
        target.append(sample[2])
        target.append(sample[2])
        # Append the different labels for original and RCM image for Adv. head
        target_jigsaw.append(sample[2])
        target_jigsaw.append(sample[3])
        # Append the ground truth patch labels
        patch_labels.append(sample[4])
        patch_labels.append(sample[5])
    # Stack the images and return the required batch for training
    return torch.stack(imgs, 0), target, target_jigsaw, patch_labels


def collate_test(batch):
    """
    Collate function for preparing the testing batch for DCL testing during training.

    :param batch: The list containing outputs of the __get_item__() function.
    Length of the list is equal to the required batch size.
    :return: The batch containing images and actual labels for testing
    """
    imgs = []  # List to store the images
    target = []  # List to store the targets
    # Iterate over each output of the __get_item__() function
    for sample in batch:
        imgs.append(sample[0])  # Append the original image
        target.append(sample[1])  # Append the origin class label
    # Stack the images and return the test batch
    return torch.stack(imgs, 0), target


class DCL:
    def __init__(self, config):
        """
        The function parses the configuration parameters and load the CUB_200_2011 dataset.

        :param config: Configuration class object
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
        """
        The function reads the train and test transformations specified in the configuration (.yml) file.
        """
        # Key to the common transforms in config
        common_transforms = self.config.cfg["dataloader"]["transforms"]["common"]
        # Key to the jigsaw transforms in config
        jigsaw_transforms = self.config.cfg["dataloader"]["transforms"]["jigsaw"]
        # Key to the train transforms in config
        train_transforms = self.config.cfg["dataloader"]["transforms"]["train"]
        # Key to the test transforms in config
        test_transforms = self.config.cfg["dataloader"]["transforms"]["test"]
        # Iterate over the common transformations in order and load them as torchvision Compose transform
        self.common_transform = transforms.Compose(
            [
                get_object_from_path(common_transforms[i]['path'])(**common_transforms[i]['param'])
                if 'param' in common_transforms[i].keys()
                else get_object_from_path(common_transforms[i]['path'])() for i in common_transforms.keys()
            ]
        )
        # Iterate over the jigsaw transformations in order and load them as torchvision Compose transform
        self.jigsaw_transform = transforms.Compose(
            [
                get_object_from_path(jigsaw_transforms[i]['path'])(**jigsaw_transforms[i]['param'])
                if 'param' in jigsaw_transforms[i].keys()
                else get_object_from_path(jigsaw_transforms[i]['path'])() for i in jigsaw_transforms.keys()
            ]
        )
        # Iterate over the train transformations in order and load them as torchvision Compose transform
        self.final_transform_train = transforms.Compose(
            [
                get_object_from_path(train_transforms[i]['path'])(**train_transforms[i]['param'])
                if 'param' in train_transforms[i].keys()
                else get_object_from_path(train_transforms[i]['path'])() for i in train_transforms.keys()
            ]
        )
        # Iterate over the test transformations in order and load them as torchvision Compose transform
        self.final_transform_test = transforms.Compose(
            [
                get_object_from_path(test_transforms[i]['path'])(**test_transforms[i]['param'])
                if 'param' in test_transforms[i].keys()
                else get_object_from_path(test_transforms[i]['path'])() for i in test_transforms.keys()
            ]
        )

    def load_dataset(self):
        """
        The function loads the train and test datasets.
        """
        # Parse configuration
        data_root_directory = self.config.cfg["dataloader"]["root_directory_path"]  # Dataset root directory path
        download = self.config.cfg["dataloader"]["download"]  # Either to download the dataset or not
        train_data_fraction = self.config.cfg["dataloader"]["train_data_fraction"]  # Fraction of dataset for training
        test_data_fraction = self.config.cfg["dataloader"]["test_data_fraction"]  # Fraction of dataset for testing
        # Jigsaw patch size for RCM
        crop_patch_size = self.config.cfg["dataloader"]["transforms"]["jigsaw"]["t_1"]["param"]["size"]
        # Prediction type for jigsaw patch prediction
        prediction_type = self.config.cfg["model"]["prediction_type"]
        # Load the train dataset
        self.train_dataset = Dataset(root=data_root_directory, train=True, download=download,
                                     crop_patch_size=crop_patch_size, common_transform=self.common_transform,
                                     jigsaw_transform=self.jigsaw_transform, final_transform=self.final_transform_train,
                                     train_data_fraction=train_data_fraction, prediction_type=prediction_type)
        # Load the test dataset
        self.test_dataset = Dataset(root=data_root_directory, train=False, download=download,
                                    final_transform=self.final_transform_test, test_data_fraction=test_data_fraction)

    def get_dataloader(self):
        """
        The function creates and returns the train and test dataloaders.

        :return: train_dataloader, test_dataloader
        """
        # Parse configuration
        batch_size = self.config.cfg["dataloader"]["batch_size"]  # Batch size for the dataloader
        shuffle = self.config.cfg["dataloader"]["shuffle"]  # Either to shuffle dataset or not
        num_workers = self.config.cfg["dataloader"]["num_workers"]  # Number of workers to load the dataset
        # Create the train dataloader
        train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, collate_fn=collate_train,
                                      shuffle=shuffle, num_workers=num_workers, pin_memory=True)
        # Create the test dataloader
        test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=batch_size, collate_fn=collate_test,
                                     num_workers=num_workers, pin_memory=True)
        # Return train and test dataloader
        return train_dataloader, test_dataloader
