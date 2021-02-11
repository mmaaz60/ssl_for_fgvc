from fine_grained_classification.dataset.cub_200_2011 import Cub2002011 as Dataset
from torch.utils.data import DataLoader


class Cub2002011:
    def __init__(self, config):
        self.data_root_directory = config.cfg["dataloader"]["root_directory_path"]
        self.resize_width = config.cfg["dataloader"]["resize_width"]
        self.resize_height = config.cfg["dataloader"]["resize_height"]
        self.batch_size = config.cfg["dataloader"]["batch_size"]
        self.shuffle = config.cfg["dataloader"]["shuffle"]
        self.num_workers = config.cfg["dataloader"]["num_workers"]
        self.download = config.cfg["dataloader"]["download"]
        self.train_dataset = None
        self.test_dataset = None
        self.load_dataset()

    def load_dataset(self):
        self.train_dataset = Dataset(root=self.data_root_directory, train=True,
                                     resize_dims=(self.resize_width, self.resize_height), download=self.download)
        self.test_dataset = Dataset(root=self.data_root_directory, train=False,
                                    resize_dims=(self.resize_width, self.resize_height), download=self.download)

    def get_dataloader(self):
        train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                                      num_workers=self.num_workers)
        test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                                     num_workers=self.num_workers)
        return train_dataloader, test_dataloader
