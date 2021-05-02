import yaml


class Configuration:
    """
    The class loads the pipeline configuration file.
    """
    cfg = None

    @classmethod
    def load_config(cls, cfg_path):
        """
        The function loads the configuration yml file into class variable cfg.
        """
        with open(cfg_path, "r") as ymlfile:
            cls.cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
