import sys


class Common:
    """
    This class initiates the trainer with specified configurations
    """
    def __init__(self, config):
        # Select the correct model
        if config.cfg["dataloader"]["name"] == "cub_200_2011":
            from fine_grained_classification.dataloader.cub_200_2011 import Cub2002011 as DataLoader
        else:
            print(f"Please provide correct dataloader to use in configuration. "
                  f"Available options are ['cub_200_2011']")
            sys.exit(1)
        # Initialize the selected DataLoader
        self.dataloader = DataLoader(config)

    def get_loader(self):
        """
        This function returns the selected dataloader
        """
        return self.dataloader.get_dataloader()
