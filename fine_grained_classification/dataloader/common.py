import sys


class Dataloader:
    """
    This class initiates the specified dataloader
    """
    def __init__(self, config):
        # Select the correct model
        if config.get("dataloader", "name") == "image_folder":
            from fine_grained_classification.dataloader.image_folder import ImageFolder as DataLoader
        else:
            print(f"Please provide correct dataloader to use in configuration. "
                  f"Available options are ['image_folder']")
            sys.exit(1)
        # Initialize the selected DataLoader
        self.dataloader = DataLoader(config)

    def get_loader(self, image, conf_thresh, nms_thresh):
        """
        This function returns the selected dataloader
        """
        return self.dataloader.get_dataloader()
