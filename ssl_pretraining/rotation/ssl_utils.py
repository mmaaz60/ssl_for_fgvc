import torch
import torch.nn as nn
from ssl_pretraining.rotation import rotation_utils as rot_utils


def preprocess_input_data(images, labels, rotation=True):
    """Preprocess a mini-batch of images."""
    if rotation:
        # Create the 4 rotated version of the images; this step increases
        # the batch size by a multiple of 4.
        batch_size_in = images.size(0)
        images = rot_utils.create_4rotations_images(images)
        labels_rotation = rot_utils.create_rotations_labels(batch_size_in, images.device)
        labels = labels.repeat(4)
    return images, labels, labels_rotation


def feature_extraction(images, feature_extractor):
    if feature_extractor == 'resnet18':
        base_feature_extractor = load_base_feature_extractor(feature_extractor)
        embedding_extractor = nn.Sequential(*list(base_feature_extractor.children())[:-1])
        features = embedding_extractor(images)
        return features


def classification_head(features, feature_extractor):
    base_feature_extractor = load_base_feature_extractor(feature_extractor)
    scores = base_feature_extractor()
    return scores
