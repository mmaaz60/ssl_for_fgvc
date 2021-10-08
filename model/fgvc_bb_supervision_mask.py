import torch.nn as nn
from utils.util import get_object_from_path
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


class AttentionResnet(nn.Module):
    """
    The class defines a specified torchvision model.
    """

    def __init__(self, config):
        """
        Constructor, the function parse the config and initialize the layers of the corresponding model,

        :param config: Configuration class object
        """
        super(AttentionResnet, self).__init__()  # Call the constructor of the parent class
        # Parse the configuration parameters
        self.model_function = get_object_from_path(config.cfg["model"]["model_function_path"])  # Model type
        self.pretrained = config.cfg["model"]["pretrained"]  # Either to load weights from pretrained model or not
        self.num_classes = config.cfg["model"]["classes_count"]  # Number of classes
        # Load the model
        net = self.model_function(pretrained=self.pretrained)
        net_list = list(net.children())
        self.feature_extractor = nn.Sequential(*net_list[:-2])  # Feature extractor
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)  # Adaptive average pooling
        self.cls_classifier = nn.Linear(in_features=net.fc.in_features, out_features=self.num_classes,
                                        bias=(net.fc.bias is not None))
        self.flatten = nn.Flatten()  # Flatten layer
        # Load the model for bbox supervision
        attention_model, _ = torch.hub.load('ashkamath/mdetr:main', 'mdetr_efficientnetB5', pretrained=True,
                                            return_postprocessor=True)
        self.attention_model = attention_model.cuda().eval()
        self.img_size = config.cfg["attention"]["augment_size"]
        self.alpha = 0.1


    def forward(self, x, train=False):
        """
        The function implements the forward pass of the model.

        :param x: Input image tensor
        :param train: Flag to specify either train or test mode
        """
        # get features of orig image
        feat_x = self.feature_extractor(x)  # Feature extraction
        feat_aug = self.attention(x, feat_x)
        feat = self.avg_pool(feat_aug)  # Adaptive average pooling
        feat = self.flatten(feat)  # Flatten the features
        # concat features
        out = self.cls_classifier(feat)
        return out

    def attention(self, x, feature_maps):
        caption = "all bird"
        bs = x.shape[0]
        captions = [caption]*bs
        memory_cache = self.attention_model(x, captions, encode_and_save=True)
        outputs = self.attention_model(x, captions, encode_and_save=False, memory_cache=memory_cache)
        # keep only top-1 prediction
        probas = 1 - outputs['pred_logits'].softmax(-1)[:, :, -1].cpu()
        _, keep = torch.max(probas, dim=1)
        _, c, h, w = feature_maps.shape
        attention_fm = feature_maps.clone()
        for b in range(bs):
            bboxes_scaled = self.rescale_bboxes((outputs['pred_boxes'].cpu()[b, keep[b]]).unsqueeze(0), (h, w))
            bboxes_scaled = torch.nn.functional.relu(bboxes_scaled)
            [x1, y1, x2, y2] = [int(box.item()) for box in bboxes_scaled[0]]
            mask = torch.ones((h, w), dtype=torch.bool)
            mask[y1:y2, x1:x2] = False
            # attention_fm[b][mask.repeat(c, 1, 1)] = attention_fm[b][mask.repeat(c, 1, 1)]*self.alpha
            attention_fm[b][mask.repeat(c, 1, 1)] *= self.alpha
        return attention_fm


    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b