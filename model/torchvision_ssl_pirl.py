import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util import get_object_from_path


class Normalize(nn.Module):
    def __init__(self, p=2):
        super(Normalize, self).__init__()
        self.p = p

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=1)


class JigsawHead(nn.Module):
    """Jigswa + linear + l2norm"""

    def __init__(self, dim_in, dim_out, k=4):
        super(JigsawHead, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, dim_out),
        )
        self.fc2 = nn.Linear(dim_out * k, dim_out)
        self.l2norm = Normalize(2)
        self.k = k

    def forward(self, x):
        bsz = x.shape[0]
        x = self.fc1(x)
        # ==== shuffle ====
        # this step can be moved to data processing step
        shuffle_ids = self.get_shuffle_ids(bsz)
        x = x[shuffle_ids]
        # ==== shuffle ====
        n_img = int(bsz / self.k)
        x = x.view(n_img, -1)
        x = self.fc2(x)
        x = self.l2norm(x)
        return x

    def get_shuffle_ids(self, bsz):
        n_img = int(bsz / self.k)
        rnd_ids = [torch.randperm(self.k) for i in range(n_img)]
        rnd_ids = torch.cat(rnd_ids, dim=0)
        base_ids = torch.arange(bsz)
        base_ids = torch.div(base_ids, self.k).long()
        base_ids = base_ids * self.k
        shuffle_ids = rnd_ids + base_ids
        return shuffle_ids


class TorchVisionSSLPIRL(nn.Module):
    """
    This class inherits from nn.Module class
    """

    def __init__(self, config):
        """
        The function parse the config and initialize the layers of the corresponding model
        :param config: YML configuration file to parse the parameters from
        """
        super(TorchVisionSSLPIRL, self).__init__()  # Call the constructor of the parent class
        # Parse the configuration parameters
        self.model_function = get_object_from_path(config.cfg["model"]["model_function_path"])  # Model type
        self.pretrained = config.cfg["model"]["pretrained"]  # Either to load weights from pretrained model or not
        self.num_classes = config.cfg["model"]["classes_count"]  # Number of classes
        # Load the model
        net = self.model_function(pretrained=self.pretrained)
        net_list = list(net.children())
        self.feature_extractor = nn.Sequential(*net_list[:-1])
        # Flatten layer
        self.flatten = nn.Flatten()
        # Classifier head
        self.cls_classifier = nn.Linear(in_features=net.fc.in_features, out_features=self.num_classes,
                                        bias=(net.fc.bias is not None))
        # MLP head for original image representation
        self.head = nn.Sequential(
            nn.Linear(net.fc.in_features, net.fc.in_features),
            nn.ReLU(inplace=True),
            nn.Linear(net.fc.in_features, 128),
            Normalize(2),
        )
        # MLP head for jigsaw image representation
        self.hed_jig = JigsawHead(dim_in=net.fc.in_features, dim_out=128)

    def forward(self, x, x_jig=None, train=True):
        """
        The function implements the forward pass of the network/model
        :param train:
        :param x: Batch of inputs (images)
        :return: The model output (class logits)
        """
        feat = self.flatten(self.feature_extractor(x))
        classification_scores = self.cls_classifier(feat)
        if train:
            feat_jig = self.flatten(self.feature_extractor(x_jig))
            jig_classification_scores = self.cls_classifier(feat_jig)
            representation = self.head(feat)
            representation_jig = self.hed_jig(feat_jig)
            return classification_scores, jig_classification_scores, representation, representation_jig
        else:
            return classification_scores
