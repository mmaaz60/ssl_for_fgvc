import torch.nn as nn
from fine_grained_classification.utils.utils import get_object_from_path
#from cam_diversification_block.diversification_block import get_CAM, diverse_block


class SSLCustomModel(nn.Module):
    """
    This class inherits from nn.Module class
    """

    def __init__(self, config):
        """
        The function parse the config and initialize the layers of the corresponding model
        :param config: YML configuration file to parse the parameters from
        """
        super(SSLCustomModel, self).__init__()  # Call the constructor of the parent class
        # Parse the configuration parameters
        self.model_function = get_object_from_path(config.cfg["SSL_model"]["model_function_path"])  # Model type
        self.pretrained = config.cfg["SSL_model"]["pretrained"]  # Either to load weights from pretrained model or not
        self.num_classes_classification = config.cfg["SSL_model"]["num_classes_classification"]   # for classifaction
        self.num_classes_rot = config.cfg["SSL_model"]["num_classes_rotation"]  # num of classes for rotation head
        self.feature_embedding = config.cfg["SSL_model"]["feature_embedding"]
        self.patch_size = config.cfg["diversification_block"]["patch_size_diversification"]
        self.alpha = config.cfg["diversification_block"]["alpha"]
        # Load the model
        self.model = self.model_function(pretrained=self.pretrained)
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=self.feature_embedding,
                                  bias=(self.model.fc.bias is not None))
        self.classification_head = nn.Linear(self.feature_embedding, self.num_classes_classification)
        self.rotation_head = nn.Linear(self.feature_embedding, self.num_classes_rot)

    def forward(self, x):
        """
        The function implements the forward pass of the network/model
        :param x: Batch of inputs (images)
        :return: The model output (class logits)
        """
        # Diverse block (...commented to check other pipeline...)
        # class_specific_maps = get_CAM(x, self.model, self.num_classes_classification)
        # gap_all_classes, activation_all_classes = diverse_block(self.model, class_specific_maps, self.patch_size, self.alpha)
        # features = self.model(activation_all_classes)
        # y_classification = self.classification_head(gap_all_classes) ( identity)
        # y_rotation = self.rotation_head(features)
        # return y_classification, y_rotation

        # Model without diverse block
        features = self.model(x)
        y_classification = self.classification_head(features)
        y_rotation = self.rotation_head(features)
        return y_classification, y_rotation
