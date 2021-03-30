import torch
from test.ssl_rot_tester import SSLROTTester
import logging
from utils.util import preprocess_input_data_rotation

logger = logging.getLogger(f"train/ssl_rot_trainer.py")


class SSLROTTrainer:
    def __init__(self, model, dataloader, class_loss_function, rot_loss_function, rotation_loss_weight, optimizer,
                 epochs, lr_scheduler=None, val_dataloader=None, device="cuda", log_step=50, checkpoints_dir_path=None,
                 diversification_test_flag=False):
        self.model = model
        self.dataloader = dataloader
        self.class_loss = class_loss_function()
        self.rot_loss = rot_loss_function()
        self.rotation_loss_weight = rotation_loss_weight
        self.optimizer = optimizer
        self.epochs = epochs
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.log_step = log_step
        self.checkpoints_dir_path = checkpoints_dir_path
        self.validator = SSLROTTester(val_dataloader, class_loss_function, rot_loss_function, model) \
            if val_dataloader else None
        self.metrics = {}

    def train_epoch(self, epoch):
        total_loss = 0
        # allocating metric for classification
        total_predictions_head1 = 0
        total_correct_predictions_head1 = 0
        # allocation metric for rotation
        total_predictions_head2 = 0
        total_correct_predictions_head2 = 0
        self.model.train()
        for batch_idx, d in enumerate(self.dataloader):
            inputs, labels = d
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            augmented_inputs, augmented_labels, rot_labels = preprocess_input_data_rotation(
                inputs, labels, rotation=True)
            class_outputs, rot_outputs = self.model(augmented_inputs)

            # computing total loss from loss for classification head and rotation head
            classification_loss = self.class_loss(class_outputs, augmented_labels)
            rot_loss = self.rot_loss(rot_outputs, rot_labels)
            loss = (1 - self.rotation_loss_weight) * classification_loss + self.rotation_loss_weight * rot_loss
            total_loss += loss

            # Metrics for classification head - head1
            _, preds_head1 = torch.max(class_outputs, 1)
            total_predictions_head1 += len(preds_head1)
            total_correct_predictions_head1 += torch.sum(preds_head1 == augmented_labels.data)
            # Metrics for rotation head - head2
            _, preds_head2 = torch.max(rot_outputs, 1)
            total_predictions_head2 += len(preds_head2)
            total_correct_predictions_head2 += torch.sum(preds_head2 == rot_labels.data)

            # optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if (batch_idx % self.log_step == 0) and (batch_idx != 0):
                logger.info(
                    f"Train Epoch: {epoch}, Step, {batch_idx}/{len(self.dataloader)}, Loss: {total_loss / batch_idx}")
        self.metrics[epoch] = {}
        self.metrics[epoch]["train"] = {}
        self.metrics[epoch]["train"]["loss"] = float(total_loss / batch_idx)
        self.metrics[epoch]["train"]["class_accuracy"] = float(
            total_correct_predictions_head1) / float(total_predictions_head1)
        self.metrics[epoch]["train"]["rot_accuracy"] = float(
            total_correct_predictions_head2) / float(total_predictions_head2)
        logger.info(f"Epoch {epoch} loss: {self.metrics[epoch]['train']['loss']}, class_accuracy:"
                    f"{self.metrics[epoch]['train']['class_accuracy']} "
                    f"rot_accuracy:{self.metrics[epoch]['train']['rot_accuracy']}")

    def train_and_validate(self, start_epoch, end_epoch=None):
        self.model = self.model.to(self.device)
        for i in range(start_epoch, end_epoch + 1 if end_epoch else self.epochs + 1):
            self.train_epoch(i)
            if self.validator:
                val_metrics = self.validator.test(self.model)
                self.metrics[i]["val"] = {}
                self.metrics[i]["val"] = val_metrics
            if self.lr_scheduler:
                self.lr_scheduler.step()
            # Save the checkpoints
            if self.checkpoints_dir_path:
                model_to_save = {
                    "epoch": i,
                    "metrics": self.metrics[i],
                    'state_dict': self.model.state_dict(),
                }
                torch.save(model_to_save,
                           f"{self.checkpoints_dir_path}/epoch_{i}.pth")
