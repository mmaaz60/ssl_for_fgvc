import sys
import os
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Add the root folder (ssl_for_fgvc) as the path
sys.path.append(f"{'/'.join(os.getcwd().split('/')[:-1])}")
from config.config import Configuration as config
from dataloader.common import Dataloader
from model.common import Model
from utils.util import get_object_from_path


class CAMVisualization:
    """
    The class implements the process of getting a cam visualization of an image for a specified model.
    """
    def __init__(self, model, model_name, cam_method='GradCAM'):
        """
        Constructor, the function initializes the class variables.

        :param model: Model to be used for the CAM visualization
        :param model_name: Model name (as per config.yml)
        :param cam_method: The method to be used for CAM calculation. Available options are
        "GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM"
        """
        self.model = model.eval()  # Put the model in the evaluation mode
        self.model_name = model_name  # The model name (as per the config.yml)
        self.cam_method = cam_method  # The cam method
        self.target_layer = None  # The target layer used to calculate the CAM
        self.cam = None  # The calculated CAM
        self._set_target_layer()  # Set the target layer as per the specified model name
        self._set_cam()  # Set cam as per the specified cam method

    def _set_target_layer(self):
        """
        The function selects the target layer as per the specified model name.
        """
        if self.model_name == "torchvision" or self.model_name == "torchvision_ssl_rotation":
            self.target_layer = self.model.model.layer4[-1]
        elif self.model_name == "torchvision_ssl_pirl":
            self.target_layer = self.model.feature_extractor[-2][-1]
        elif self.model_name == "dcl":
            self.target_layer = self.model.feature_extractor[-1][-1]
        else:
            print(f"Given model ({self.model_name}) is not supported. Exiting!")
            sys.exit(1)

    def _set_cam(self):
        """
        The function selects the cam visualization method specified by cam_method
        """
        if self.cam_method == "GradCAM":
            self.cam = GradCAM(model=self.model, target_layer=self.target_layer, use_cuda=True)
        elif self.cam_method == "GradCAMPlusPlus":
            self.cam = GradCAMPlusPlus(model=self.model, target_layer=self.target_layer, use_cuda=True)
        elif self.cam_method == "ScoreCAM":
            self.cam = ScoreCAM(model=self.model, target_layer=self.target_layer, use_cuda=True)
        elif self.cam_method == "AblationCAM":
            self.cam = AblationCAM(model=self.model, target_layer=self.target_layer, use_cuda=True)
        elif self.cam_method == "XGradCAM":
            self.cam = XGradCAM(model=self.model, target_layer=self.target_layer, use_cuda=True)
        else:
            self.cam = GradCAM(model=self.model, target_layer=self.target_layer, use_cuda=True)

    def get_cam_image(self, x, x_orig):
        """
        The function interpolates the class activation maps and return an image of required size

        :param x: Batch of images (b, c, h, w)
        """
        grayscale_cam = self.cam(input_tensor=x, target_category=1)
        visualization = show_cam_on_image(np.array(x_orig, dtype=np.float32) / 255.0, grayscale_cam, use_rgb=True)
        pil_image = Image.fromarray(visualization)
        # Get the classification label
        cls_scores = self.model(x)
        _, label = torch.max(cls_scores, 1)

        return pil_image, int(label.detach())


def parse_arguments():
    """
    Parse the command line arguments
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-config", "--config_path", required=True,
                    help="The path to the pipeline .yml configuration file.")
    ap.add_argument("-cam", "--cam_method", required=False, default='GradCAM',
                    help="Cam method to use. Possible options are "
                         "[GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM]")
    ap.add_argument("-checkpoints", "--model_checkpoints", required=True,
                    help="The path to model checkpoints.")
    ap.add_argument("-dataset", "--root_dataset_path", required=False, default="./data/CUB_200_2011",
                    help="The path to the dataset root directory. "
                         "The program will download the dataset if not present locally.")
    ap.add_argument("-save", "--output_directory", required=True,
                    help="The path to output directory to save the visualizations.")
    ap.add_argument("-dim", "--output_dim", type=int, required=False, default=448,
                    help="The output dimensions of the images overlayed with CAMs.")
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
    if not os.path.exists(f"{args['output_directory']}/correct_predictions"):
        os.mkdir(f"{args['output_directory']}/correct_predictions")
    if not os.path.exists(f"{args['output_directory']}/wrong_predictions"):
        os.mkdir(f"{args['output_directory']}/wrong_predictions")
    config.load_config(args["config_path"])  # Load configuration
    config.cfg["dataloader"]["root_directory_path"] = args["root_dataset_path"]  # Set the dataset path
    _, test_loader = Dataloader(config=config).get_loader()  # Create dataloader
    # Get the required attributes from the dataset
    data = test_loader.dataset.data.values
    data = data[np.argsort(data[:, 0])]
    image_ids = data[:, 0]
    test_image_paths = data[:, 1]
    test_image_labels = data[:, 2]
    # Create the model
    model = Model(config=config).get_model()
    model = model.to(args["device"])
    # Load pretrained weights
    checkpoints_path = args["model_checkpoints"]
    checkpoints = torch.load(checkpoints_path)
    model.load_state_dict(checkpoints["state_dict"], strict=True)
    # Create CAM visualizer object
    visualizer = CAMVisualization(model, config.cfg["model"]["name"], cam_method=args["cam_method"])
    # Create transforms for performing inference
    resize_dim = (config.cfg["dataloader"]["resize_width"], config.cfg["dataloader"]["resize_height"])
    infer_dim = args["output_dim"]
    test_transforms = config.cfg["dataloader"]["transforms"]["test"]
    test_transform = transforms.Compose(
        [
            get_object_from_path(test_transforms[i]['path'])(**test_transforms[i]['param'])
            if 'param' in test_transforms[i].keys()
            else get_object_from_path(test_transforms[i]['path'])() for i in test_transforms.keys()
        ]
    )
    # Iterate over the dataset
    for image_info in zip(image_ids, test_image_paths, test_image_labels):
        image_id, image_path, image_label = image_info
        full_path = os.path.join(config.cfg["dataloader"]["root_directory_path"],
                                 "CUB_200_2011/images", image_path)
        input = Image.open(full_path).convert('RGB')
        input = input.resize(resize_dim, Image.ANTIALIAS)
        input_trans = test_transform(input)  # Transform the image
        input_trans = torch.unsqueeze(input_trans, 0)
        input_trans = input_trans.to(args["device"])
        # Get the cam image
        output_image, predicted_label = visualizer.get_cam_image(input_trans,
                                                                 input.resize((infer_dim, infer_dim), Image.ANTIALIAS))
        # Write the cam images to the disc
        predicted_label += 1
        if predicted_label == image_label:
            # Save the PIL image
            output_image.save(f"{args['output_directory']}/correct_predictions/"
                              f"{image_id}_{image_label}_{predicted_label}.jpg")
        else:
            # Save the PIL image
            output_image.save(f"{args['output_directory']}/wrong_predictions/"
                              f"{image_id}_{image_label}_{predicted_label}.jpg")


if __name__ == "__main__":
    main()
