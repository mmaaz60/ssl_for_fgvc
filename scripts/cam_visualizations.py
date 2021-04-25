import argparse
import torch
from torch import nn
import numpy as np
from PIL import Image
from matplotlib import cm


class CAMVisualization:
    def __init__(self, model):
        self.model = model

    def _get_cam(self, x):
        out, cam = self.model.get_cam(x)
        return out, cam

    def get_cam_image(self, cam, dim, h, w):
        """
        The function interpolates the class activation maps and return an image of required size
        """
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
        pass

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
    ap.add_argument("-root", "--cub_root_directory", required=True,
                    help="The root directory for CUB dataset.")
    ap.add_argument("-m", "--model", required=True,
                    help="The model name to be used for inference.")
    ap.add_argument("-weights", "--trained_weights_path", required=True,
                    help="Path to the trained model weights.")
    ap.add_argument("-save", "--output_directory", required=True,
                    help="The path to output directory to save the visualizations.")
    args = vars(ap.parse_args())

    return args


def main():
    """
    Implements the main flow, i.e. load the dataset & model, generate cam visualizations and save the visualizations
    """
    args = parse_arguments()
    pass


if __name__ == "__main__":
    main()
