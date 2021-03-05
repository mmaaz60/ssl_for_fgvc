import sys
import torch
import torchvision
import torchvision.transforms as transforms

class Dataloader:
    """
    This class initiates the specified dataloader. This dataloader can be used to verify the performance of SSL on
    Classification problems independent of FGC
    """
    def __init__(self, config):

        if config.cfg["dataloader"]["name"] == "CIFAR10":
            """
            Get the configs for the CIFAR10 dataset
            """
            self.batch_size = config.cfg["dataloader"]["batch_size"]  # Batch size for the dataloader
            self.shuffle = config.cfg["dataloader"]["shuffle"]  # Either to shuffle dataset or not
            self.num_workers = config.cfg["dataloader"]["num_workers"]  # Number of workers to load the dataset
            self.download = config.cfg["dataloader"]["download"]  # Either to download the dataset or not

            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            image_datasets = {'train': torchvision.datasets.CIFAR10(root='./data', train=True,
                                                                    download=True, transform=transform),
                              'test': torchvision.datasets.CIFAR10(root='./data', train=False,
                                                                  download=True, transform=transform)}
            dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
            dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                                          shuffle=True, num_workers=2)
                           for x in ['train', 'test']}
        else:
            print(f"Please provide correct dataloader to use in configuration. "
                  f"Available options are ['cub_200_2011']")
            sys.exit(1)
        # Initialize the selected DataLoader
        self.dataloader = dataloaders

    def get_loader(self):
        """
        This function returns the selected dataloader
        """
        train_dataloader = self.dataloader['train']
        test_dataloader = self.dataloader['test']
        return train_dataloader, test_dataloader

