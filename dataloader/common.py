import sys


class Dataloader:
    """
    This class initiates the specified dataloader
    """
    def __init__(self, config):
        # Select the correct model
        if config.cfg["dataloader"]["name"] == "cub_200_2011":
            from dataloader.cub_200_2011 import Cub2002011 as DataLoader
        elif config.cfg["dataloader"]["name"] == "cub_200_2011_contrastive":
            from dataloader.cub_200_2011_contrastive import Cub2002011Contrastive as DataLoader
        else:
            print(f"Please provide correct dataloader to use in configuration. "
                  f"Available options are ['cub_200_2011', 'cub_200_2011_contrastive']")
            sys.exit(1)
        # Initialize the selected DataLoader
        self.dataloader = DataLoader(config)

    def get_loader(self):
        """
        This function returns the selected dataloader
        """
        return self.dataloader.get_dataloader()
