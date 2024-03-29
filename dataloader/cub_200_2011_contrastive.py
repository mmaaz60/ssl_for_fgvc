from dataset.cub_200_2011_contrastive import Cub2002011Contrastive as Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.util import get_object_from_path


class Cub2002011Contrastive:
    """
    The class defines the flow of loading the dataloaders for CUB-200-2011 dataset for contrastive SSL training.
    """
    def __init__(self, config):
        """
        Constructor, the function parse the configuration parameters and load the CUB_200_2011 dataset.

        :param config: Configuration class object
        """
        self.config = config
        self.train_transform = None  # Train transform
        self.test_transform = None  # Test transform
        self.contrastive_transforms = None  # Contrastive transforms
        self.train_dataset = None  # Train dataset
        self.test_dataset = None  # Test Dataset
        self.get_transforms()  # Parse the train and test transforms from configuration file
        self.load_dataset()  # Load the train and test datasets

    def get_transforms(self):
        """
            The function reads the train and test transformations specified in the configuration (.yml) file.
        """
        train_transforms = self.config.cfg["dataloader"]["transforms"]["train"]  # Key to the train transforms in config
        test_transforms = self.config.cfg["dataloader"]["transforms"]["test"]  # Key to the test transforms in config
        # Iterate over the train transformations in order and load them as torchvision Compose transform
        self.train_transform = transforms.Compose(
            [
                get_object_from_path(train_transforms[i]['path'])(**train_transforms[i]['param'])
                if 'param' in train_transforms[i].keys()
                else get_object_from_path(train_transforms[i]['path'])() for i in train_transforms.keys()
            ]
        )
        # Iterate over the test transformations in order and load them as torchvision Compose transform
        self.test_transform = transforms.Compose(
            [
                get_object_from_path(test_transforms[i]['path'])(**test_transforms[i]['param'])
                if 'param' in test_transforms[i].keys()
                else get_object_from_path(test_transforms[i]['path'])() for i in test_transforms.keys()
            ]
        )
        # Load the transformations for contrastive self-supervised learning
        self.contrastive_transforms = get_object_from_path(self.config.cfg["dataloader"]["transforms"]["contrastive"])

    def load_dataset(self):
        """
        The function loads the train and test datasets.
        """
        # Parse configuration
        data_root_directory = self.config.cfg["dataloader"]["root_directory_path"]  # Dataset root directory path
        resize_width = self.config.cfg["dataloader"]["resize_width"]  # Image resize width
        resize_height = self.config.cfg["dataloader"]["resize_height"]  # Image resize height
        download = self.config.cfg["dataloader"]["download"]  # Either to download the dataset or not
        train_data_fraction = self.config.cfg["dataloader"]["train_data_fraction"]  # Fraction of dataset for training
        test_data_fraction = self.config.cfg["dataloader"]["test_data_fraction"]  # Fraction of dataset for testing
        # Load the train dataset
        self.train_dataset = Dataset(root=data_root_directory, train=True, resize_dims=(resize_width, resize_height),
                                     transform=self.train_transform, contrastive_transforms=self.contrastive_transforms,
                                     download=download, train_data_fraction=train_data_fraction)
        # Load the test dataset
        self.test_dataset = Dataset(root=data_root_directory, train=False, resize_dims=(resize_width, resize_height),
                                    transform=self.test_transform, download=download,
                                    test_data_fraction=test_data_fraction)

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
        train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=shuffle,
                                      num_workers=num_workers)
        # Create the test dataloader
        test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=batch_size, shuffle=shuffle,
                                     num_workers=num_workers)
        # Return train and test dataloader
        return train_dataloader, test_dataloader
