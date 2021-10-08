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
        self.model = self.model_function(pretrained=self.pretrained)
        # Alter the classification layer as per the specified number of classes
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=self.num_classes,
                                  bias=(self.model.fc.bias is not None))
        attention_model, _ = torch.hub.load('ashkamath/mdetr:main', 'mdetr_efficientnetB5', pretrained=True,
                                              return_postprocessor=True)
        self.attention_model = attention_model.cuda().eval()
        self.img_size = config.cfg["attention"]["augment_size"]

    def forward(self, x, train=False):
        """
        The function implements the forward pass of the model.

        :param x: Input image tensor
        :param train: Flag to specify either train or test mode
        """
        x_aug = self.attention(x)
        out = self.model(x_aug)
        return out

    def attention(self, x):
        caption = "all bird"
        bs = x.shape[0]
        captions = [caption]*bs
        memory_cache = self.attention_model(x, captions, encode_and_save=True)
        outputs = self.attention_model(x, captions, encode_and_save=False, memory_cache=memory_cache)
        # keep only top-1 prediction
        probas = 1 - outputs['pred_logits'].softmax(-1)[:, :, -1].cpu()
        _, keep = torch.max(probas, dim=1)
        x_aug = torch.zeros_like(x)
        for b in range(bs):
            bboxes_scaled = self.rescale_bboxes((outputs['pred_boxes'].cpu()[b, keep[b]]).unsqueeze(0),
                                                (self.img_size, self.img_size))
            bboxes_scaled = torch.nn.functional.relu(bboxes_scaled)
            [x1, y1, x2, y2] = [int(box.item()) for box in bboxes_scaled[0]]
            x_aug[b, :] = F.interpolate(x[b, None, :, y1:y2, x1:x2], size=(self.img_size, self.img_size))

        return x_aug

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
