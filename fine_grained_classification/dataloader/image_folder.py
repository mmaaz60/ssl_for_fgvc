from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

# Use help from the below code if you want to normalize the data using mean and standard deviation
# means = (0.485, 0.456, 0.406)
# stds = (0.229, 0.224, 0.225)
# TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(means, stds)])


class ImageFolder:
    def __init__(self, config):
        self.data_root_directory = config.cfg["dataloader", "root_directory_path"]
        self.resize_width = config.cfg["dataloader", "resize_width"]
        self.resize_height = config.cfg["dataloader", "resize_height"]
        self.batch_size = config.cfg["dataloader", "batch_size"]
        self.shuffle = config.cfg["dataloader", "shuffle"]
        self.num_workers = config.cfg["dataloader", "num_workers"]
        self.dataset = datasets.ImageFolder(root=self.data_root_directory,
                                            transform=transforms.Compose([transforms.ToTensor()]),
                                            loader=self.image_loader)

    def image_loader(self, path):
        image = Image.open(path)
        image = image.resize((self.resize_width, self.resize_height))

        return image

    def get_dataloader(self):
        return DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers)
