import torch
from test.base_tester import BaseTester
import logging
from utils.util import preprocess_input_data_rotation
from utils.util import save_model_checkpoints

logger = logging.getLogger(f"train/ssl_rot_trainer.py")


class SSLROTTrainer:
    def __init__(self, model, dataloader, class_loss_function, rot_loss_function, rotation_loss_weight, optimizer,
                 epochs, lr_scheduler=None, val_dataloader=None, device="cuda", log_step=50, checkpoints_dir_path=None):
        """
        Constructor, the function initializes the training related parameters.

        :param model: The model to train
        :param dataloader: The dataloader to get training samples from
        :param class_loss_function:  The CUB classification loss function
        :param rot_loss_function: The rotation classification loss function
        :param rotation_loss_weight: The lambda value, specifying the contribution of rotation loss
        :param optimizer: The optimizer to be used for training
        :param epochs: Number of epochs
        :param lr_scheduler:  Learning rate scheduler
        :param val_dataloader: Validation dataloader to get the validation samples
        :param device:  The execution device
        :param log_step:  # Logging step, after each log_step batches a log will be recorded
        :param checkpoints_dir_path:  # Checkpoints directory to save the model training progress and checkpoints
        """
        self.model = model
        self.dataloader = dataloader
        self.class_loss = class_loss_function()
        self.rot_loss = rot_loss_function()
        self.rotation_loss_weight = rotation_loss_weight  # Decides contribution of rotation loss to total loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.log_step = log_step
        self.checkpoints_dir_path = checkpoints_dir_path
        self.validator = BaseTester(val_dataloader, class_loss_function) \
            if val_dataloader else None
        self.metrics = {}

    def train_epoch(self, epoch):
        """
        The function trains the model for one epoch.
        """
        total_cls_loss = 0
        total_rot_loss = 0
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
            # Generates rotation augmented images and corresponding labels
            # Augmented labels: Repeats of original class labels for each rotation of image
            augmented_inputs, augmented_labels, rot_labels = preprocess_input_data_rotation(
                inputs, labels, rotation=True)
            class_outputs, rot_outputs = self.model(augmented_inputs, train=True)

            # Computing total loss from loss for classification head and rotation head
            classification_loss = self.class_loss(class_outputs, augmented_labels)
            total_cls_loss += classification_loss
            rot_loss = self.rot_loss(rot_outputs, rot_labels)
            total_rot_loss += rot_loss
            # Limits contribution of rotation loss by rotation_loss_weight
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
                    f"Train Epoch: {epoch}, Step, {batch_idx}/{len(self.dataloader)}, "
                    f"Cls Loss: {total_cls_loss / batch_idx}, Rot Loss: {total_rot_loss / batch_idx} "
                    f"Total Loss: {total_loss / batch_idx}")
        self.metrics[epoch] = {}
        self.metrics[epoch]["train"] = {}
        self.metrics[epoch]["train"]["loss"] = float(total_loss / batch_idx)
        self.metrics[epoch]["train"]["cls_loss"] = float(total_cls_loss / batch_idx)
        self.metrics[epoch]["train"]["rot_loss"] = float(total_rot_loss / batch_idx)
        self.metrics[epoch]["train"]["class_accuracy"] = float(
            total_correct_predictions_head1) / float(total_predictions_head1)
        self.metrics[epoch]["train"]["rot_accuracy"] = float(
            total_correct_predictions_head2) / float(total_predictions_head2)
        logger.info(f"Epoch {epoch} cls loss: {self.metrics[epoch]['train']['cls_loss']}, "
                    f"Epoch {epoch} rot loss: {self.metrics[epoch]['train']['rot_loss']}, "
                    f"Epoch {epoch} loss: {self.metrics[epoch]['train']['loss']}, "
                    f"class_accuracy:{self.metrics[epoch]['train']['class_accuracy']} "
                    f"rot_accuracy:{self.metrics[epoch]['train']['rot_accuracy']}")

    def train_and_validate(self, start_epoch, end_epoch=None):
        """
        The function implements the overall training pipeline.

        :param start_epoch: Start epoch number
        :param end_epoch: End epoch number
        """
        self.model = self.model.to(self.device)  # Transfer the model to the execution device
        best_accuracy = 0  # Variable to keep track of the best test accuracy to save the best model
        # Train and validate the model for (end_epoch - start_epoch)
        for i in range(start_epoch, end_epoch + 1 if end_epoch else self.epochs + 1):
            self.train_epoch(i)
            if self.validator:
                val_metrics = self.validator.test(self.model)
                self.metrics[i]["val"] = {}
                self.metrics[i]["val"] = val_metrics
                # Save the checkpoints
                best_accuracy = save_model_checkpoints(self.checkpoints_dir_path, i, self.model.state_dict(),
                                                       self.metrics[i], best_accuracy)
            if self.lr_scheduler:
                self.lr_scheduler.step()
