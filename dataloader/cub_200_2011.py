from dataset.cub_200_2011 import Cub2002011 as Dataset
from torch.utils.data import DataLoader


class Cub2002011:
    def __init__(self, config):
        """
        The function parse the configuration parameters and load the CUB_200_2011 dataset
        :param config: YML configuration object
        """
        self.data_root_directory = config.cfg["dataloader"]["root_directory_path"]  # Dataset root directory path
        self.resize_width = config.cfg["dataloader"]["resize_width"]  # Image resize width
        self.resize_height = config.cfg["dataloader"]["resize_height"]  # Image resize height
        self.batch_size = config.cfg["dataloader"]["batch_size"]  # Batch size for the dataloader
        self.shuffle = config.cfg["dataloader"]["shuffle"]  # Either to shuffle dataset or not
        self.num_workers = config.cfg["dataloader"]["num_workers"]  # Number of workers to load the dataset
        self.download = config.cfg["dataloader"]["download"]  # Either to download the dataset or not
        self.train_dataset = None  # Train dataset
        self.test_dataset = None  # Test Dataset
        self.load_dataset()  # Load the train and test datasets

    def load_dataset(self):
        """
        Load the train and test datasets
        """
        # Load the train dataset
        self.train_dataset = Dataset(root=self.data_root_directory, train=True,
                                     resize_dims=(self.resize_width, self.resize_height), download=self.download)
        # Load the test dataset
        self.test_dataset = Dataset(root=self.data_root_directory, train=False,
                                    resize_dims=(self.resize_width, self.resize_height), download=self.download)

    def get_dataloader(self):
        """
        Create and return train and test dataloaders
        :return: train_dataloader, test_dataloader
        """
        # Create the train dataloader
        train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                                      num_workers=self.num_workers)
        # Create the test dataloader
        test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                                     num_workers=self.num_workers)
        # Return train and test dataloader
        return train_dataloader, test_dataloader
