import torch
import logging
from utils.util import preprocess_input_data_rotation

logger = logging.getLogger(f"test/ssl_rot_tester.py")


class SSLROTTester:
    def __init__(self, dataloader, class_loss_function, rot_loss_function, model, device="cuda"):
        self.dataloader = dataloader
        self.class_loss = class_loss_function()
        self.rot_loss = rot_loss_function()
        self.device = device
        self.model = model

    def test(self, model):
        metrics = {}
        model.eval()
        with torch.no_grad():
            total_loss = 0
            # allocating metric for classification
            total_predictions_head1 = 0
            total_correct_predictions_head1 = 0
            # allocation metric for rotation
            total_predictions_head2 = 0
            total_correct_predictions_head2 = 0

            for batch_idx, d in enumerate(self.dataloader):
                inputs, labels = d
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                augmented_inputs, augmented_labels, rot_labels = preprocess_input_data_rotation(
                    inputs, labels, rotation=True)
                class_outputs, rot_outputs = model(augmented_inputs, train=False)

                # computing total loss from loss for classification head and rotation head
                classification_loss = self.class_loss(class_outputs, augmented_labels)
                rot_loss = self.rot_loss(rot_outputs, rot_labels)
                loss = classification_loss + rot_loss
                total_loss += loss

                # Metrics for classification head - head1
                _, preds_head1 = torch.max(class_outputs, 1)
                total_predictions_head1 += len(preds_head1)
                total_correct_predictions_head1 += torch.sum(preds_head1 == augmented_labels.data)
                # Metrics for rotation head - head2
                _, preds_head2 = torch.max(rot_outputs, 1)
                total_predictions_head2 += len(preds_head2)
                total_correct_predictions_head2 += torch.sum(preds_head2 == rot_labels.data)

            metrics['loss'] = float(total_loss) / len(self.dataloader)
            metrics['class_accuracy'] = float(total_correct_predictions_head1) / total_predictions_head1
            metrics['rot_accuracy'] = float(total_correct_predictions_head2) / total_predictions_head2

            logger.info(f"Validation loss: {metrics['loss']}, Validation class_accuracy: {metrics['class_accuracy']}, "
                        f"Validation rot_accuracy: {metrics['rot_accuracy']}")

        return metrics
