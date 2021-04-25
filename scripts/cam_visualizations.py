import sys
import os
import argparse
import torch
from torch import nn
import numpy as np
from PIL import Image
from matplotlib import cm
from torchvision import transforms

# Add the root folder (ssl_for_fgvc) as the path
sys.path.append(f"{'/'.join(os.getcwd().split('/')[:-1])}")
from config.config import Configuration as config
from dataloader.common import Dataloader
from model.common import Model
from utils.util import get_object_from_path


class CAMVisualization:
    def __init__(self, model):
        self.model = model
        self.cam = None

    def get_cam_image(self, x, x_orig):
        """
        The function interpolates the class activation maps and return an image of required size
        :param x: Batch of images (b, c, h, w)
        """
        b, c, h, w = x.shape
        cam = self.model.get_cam(x)
        print(cam.shape)
        return x_orig

        topk = 1
        min_val, min_args = torch.min(cam, dim=dim, keepdim=True)
        cam -= min_val
        max_val, max_args = torch.max(cam, dim=dim, keepdim=True)
        cam /= max_val
        topk_cam = cam.view(1, -1, h, w)[0, topk]
        cams = nn.functional.interpolate(topk_cam.unsqueeze(0), (h, w), mode='bilinear', align_corners=True).squeeze(0)
        topk_cam = torch.split(topk_cam, 1)
        # cams = topk_cam[k].squeeze().cpu().data.numpy() if k top cams to identify
        cam_ = topk_cam.squeeze().cpu().data.numpy()
        cam_pil = array_to_cam(cam_)
        # blended_cam = blend(img_pil, cam_pil)
        return cam_pil


def array_to_cam(arr):
    cam_pil = Image.fromarray(np.uint8(cm.gist_earth(arr) * 255)).convert("RGB")
    return cam_pil


def blend(image1, image2, alpha=0.75):
    return Image.blend(image1, image2, alpha)


def parse_arguments():
    """
    Parse the command line arguments
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-config", "--config_path", required=True,
                    help="The path to the pipeline .yml configuration file.")
    ap.add_argument("-save", "--output_directory", required=True,
                    help="The path to output directory to save the visualizations.")
    ap.add_argument("-d", "--device", required=False, default='cuda',
                    help="The computation device to perform operations ('cpu', 'cuda')")

    args = vars(ap.parse_args())

    return args


def main():
    """
    Implements the main flow, i.e. load the dataset & model, generate cam visualizations and save the visualizations
    """
    args = parse_arguments()  # Parse arguments
    # Create the output directory if not exists
    if not os.path.exists(args["output_directory"]):
        os.makedirs(args["output_directory"])
    config.load_config(args["config_path"])  # Load configuration
    _, test_loader = Dataloader(config=config).get_loader()  # Create dataloader
    test_image_paths = test_loader.dataset.data.values[:, 1]
    # Create the model
    model = Model(config=config).get_model()
    model = model.to(args["device"])
    # Load pretrained weights
    checkpoints_path = config.cfg["model"]["checkpoints_path"]
    checkpoints = torch.load(checkpoints_path)
    model.load_state_dict(checkpoints["state_dict"], strict=True)
    # Create CAM visualizer object
    visualizer = CAMVisualization(model)
    tensor_to_pil = transforms.ToPILImage()  # Create transform object to covert torch tensor to PIL image
    # Create transforms for performing inference
    test_transforms = config.cfg["dataloader"]["transforms"]["test"]
    test_transform = transforms.Compose(
        [
            get_object_from_path(test_transforms[i]['path'])(**test_transforms[i]['param'])
            if 'param' in test_transforms[i].keys()
            else get_object_from_path(test_transforms[i]['path'])() for i in test_transforms.keys()
        ]
    )
    # Iterate over the dataset
    for i, image_path in enumerate(test_image_paths):
        full_path = os.path.join(config.cfg["dataloader"]["root_directory_path"],
                                 "CUB_200_2011/images", image_path)
        input = Image.open(full_path)  # Read the image
        input_trans = test_transform(input)  # Transform the image
        input_trans = torch.unsqueeze(input_trans, 0)
        input_trans = input_trans.to(args["device"])
        output_image = visualizer.get_cam_image(input_trans, input)  # Get the cam image
        # Write the cam images to the disc
        output_image.save(f"{args['output_directory']}/{i}.jpg")  # Save the PIL image


if __name__ == "__main__":
    main()
