import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torchvision import transforms
from torch.utils.data import Dataset
import tarfile
from PIL import Image
import torch


class Cub2002011(Dataset):
    base_folder = 'CUB_200_2011/images'  # Base dataset path
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'  # Dataset download URL
    filename = 'CUB_200_2011.tgz'  # Dataset TGZ file name
    tgz_md5 = '97eceeb196236b17998738112f37df78'  # MD5 signature for the downloaded dataset TGZ file

    def __init__(self, root, train=True, download=True, loader=default_loader, resize_dims=None,
                 mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        """
        Initialize the class variables, download the dataset (if prompted to do so), verify the data presence,
        and load the dataset metadata.
        :param root: Dataset root path
        :param train: Train dataloader flag (True: Train Dataloader, False: Test Dataloader)
        :param download: Flag set to download the dataset
        :param loader: The data point loader function (By default a PyTorch default_loader(PIL loader) is used)
        :param resize_dims: Image resize dimensions
        """
        self.root = os.path.expanduser(root)
        self.train = train
        self.loader = loader
        self.resize_dims = resize_dims
        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        if train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ]
            )

    @staticmethod
    def scale_keep_ar_min_fixed(img, fixed_min):
        ow, oh = img.size

        if ow < oh:

            nw = fixed_min

            nh = nw * oh // ow

        else:

            nh = fixed_min

            nw = nh * ow // oh
        return img.resize((nw, nh), Image.BICUBIC)

    def _load_metadata(self):
        """
        Load the metadata (image list, class list and train/test split) of the CUB_200_2011 dataset
        """
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        """
        Verify if the dataset is present and loads accurately
        """
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        """
        Download the CUB_200_2011 dataset if not downloaded already
        """
        # Check if the files are already downloaded
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        # Download the data if not downloaded already
        download_url(self.url, self.root, self.filename, self.tgz_md5)
        # Extract the downloaded dataset
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        """
        The function overrides the __len__ method of Dataset class
        :return: Length of the CUB dataset (train/test)
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        The function overrides the __getitem__ method of the Dataset class
        :param idx: The index to fetch the data entry/sample
        :return: The image tensor and corresponding label
        """
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)  # Call the loader function to load the image
        # Resize the image if the resize dims are specified
        if self.resize_dims is not None:
            img = img.resize(self.resize_dims)
        # Apply the transforms if the transformations are specified
        if self.transform is not None:
            img = self.transform(img)
        # Return the image tensor and corresponding target/label
        return img, target
