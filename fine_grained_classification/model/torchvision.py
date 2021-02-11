import torch.nn as nn
import torchvision.models as models


class TorchVision(nn.Module):
    def __init__(self, config):
        super(TorchVision, self).__init__()
        self.type = config.cfg["model"]["type"]
        self.pretrained = config.cfg["model"]["pretrained"]
        self.num_classes = config.cfg["model"]["classes_count"]
        self.model = self.select_model()
        modules = list(self.model.children())[:-1]
        self.features = nn.Sequential(*modules)
        self.linear = nn.Linear(self.model.fc.in_features, self.num_classes)
        self.softmax = nn.Softmax(dim=1)

    def select_model(self):
        if self.type == "resnet50":
            return models.resnet50(pretrained=self.pretrained, progress=True)
        else:
            raise RuntimeError(f"The requested model type is not support. The supported model types are ['resnet50']")

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.softmax(out)
        return out
