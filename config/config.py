import yaml


class Configuration:
    cfg = None

    @classmethod
    def load_config(cls, cfg_path):
        with open(cfg_path, "r") as ymlfile:
            cls.cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
