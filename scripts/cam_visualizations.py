import argparse


class CAMVisualization:
    def __init__(self, model):
        self.model = model

    def _get_cam(self, x):
        return self.model.get_cam(x)

    def get_cam_image(self, dims):
        """
        The function interpolates the class activation maps and return an image of required size
        """
        pass


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
