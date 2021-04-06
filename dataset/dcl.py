import os
from torchvision.datasets.folder import default_loader
from dataset.cub_200_2011 import Cub2002011
from utils.util import get_image_crops
from PIL import ImageStat


class DCL(Cub2002011):
    def __init__(self, root, train=True, download=True, crop_patch_size=(7, 7), num_classes=200, common_transform=None,
                 jigsaw_transform=None, final_transform=None, train_data_fraction=1, test_data_fraction=1,
                 class_type=None):
        """
        Initialize the class variables, download the dataset (if prompted to do so), verify the data presence,
        :param root: Dataset root path
        :param train: Train dataloader flag (True: Train Dataloader, False: Test Dataloader)
        :param download: Flag set to download the dataset
        """
        super().__init__(root=root, train=train, download=download, loader=default_loader, resize_dims=None,
                         transform=None, train_data_fraction=train_data_fraction, test_data_fraction=test_data_fraction)
        self.num_classes = num_classes
        self.crop_patch_size = crop_patch_size
        self.common_transform = common_transform
        self.jigsaw_transform = jigsaw_transform
        self.final_transform = final_transform
        self.class_type = class_type

    def __getitem__(self, idx):
        """
        The function overrides the __getitem__ method of the Cub2002011 class
        :param idx: The index to fetch the data entry/sample
        :return: The image tensor and corresponding label
        """
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)  # Call the loader function to load the image
        if self.train:
            img_original = self.common_transform(img) if self.common_transform is not None else img
            img_original_list = get_image_crops(img_original, self.crop_patch_size)
            original_patch_range = self.crop_patch_size[0] * self.crop_patch_size[1]
            original_patch_labels = [(i - (original_patch_range // 2)) / original_patch_range
                                     for i in range(original_patch_range)]
            original_patch_labels_cls = list(range(1, original_patch_range + 1))
            img_jigsaw, jigsaw_ind = self.jigsaw_transform(img) if self.jigsaw_transform is not None else img
            img_jigsaw_list = get_image_crops(img_jigsaw, self.crop_patch_size)
            original_stats = [sum(ImageStat.Stat(im).mean) for im in img_original_list]
            jigsaw_stats = [sum(ImageStat.Stat(im).mean) for im in img_jigsaw_list]
            jigsaw_patch_labels = []
            for jigsaw_stat in jigsaw_stats:
                distance = [abs(jigsaw_stat - original_stat) for original_stat in original_stats]
                index = distance.index(min(distance))
                jigsaw_patch_labels.append(original_patch_labels[index])
            # Creating labels from tracked jigsaw_ind
            jigsaw_patch_labels_ind = []
            for i in range(original_patch_range):
                jigsaw_patch_labels_ind.append(original_patch_labels[jigsaw_ind[i]-1])
            img_jigsaw = self.final_transform(img_jigsaw) if self.final_transform is not None else img_jigsaw
            target_jigsaw = target + self.num_classes
            img_original = self.final_transform(img_original) if self.final_transform is not None else img_original
            if self.class_type == "class":
                return img_original, img_jigsaw, target, target_jigsaw, original_patch_labels_cls, jigsaw_ind
            else:
                return img_original, img_jigsaw, target, target_jigsaw, original_patch_labels, jigsaw_patch_labels_ind
        else:
            # Apply the transforms if the transformations are specified
            if self.final_transform is not None:
                img = self.final_transform(img)
            # Return the image tensor and corresponding target/label
            return img, target
