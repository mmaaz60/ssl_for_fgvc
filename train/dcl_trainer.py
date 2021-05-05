import torch
from test.dcl_tester import DCLTester
import logging
from torch.autograd import Variable
import numpy as np
from utils.util import save_model_checkpoints

logger = logging.getLogger(f"train/dcl_trainer.py")


class DCLTrainer:
    def __init__(self, model, dataloader, cls_loss_function, adv_loss_function, jigsaw_loss_function, use_adv,
                 use_jigsaw, optimizer, epochs, lr_scheduler=None, test_dataloader=None, device="cuda", log_step=50,
                 checkpoints_dir_path=None):
        """
        Constructor, the function initializes the training related parameters.

        :param model: The model to train
        :param dataloader: The dataloader to get training samples from
        :param cls_loss_function:  The classification loss function
        :param adv_loss_function: The adversarial loss function
        :param jigsaw_loss_function: The loss function for jigsaw permutation prediction
        :param use_adv: Flag to decide if to use Adv loss to calculate the overall loss or not
        :param use_jigsaw: Flag to decide if to use jigsaw loss to calculate the overall loss or not
        :param optimizer: The optimizer to be used for training
        :param epochs: Number of epochs
        :param lr_scheduler:  Learning rate scheduler
        :param test_dataloader: Test dataloader to get the test samples
        :param device:  The execution device
        :param log_step:  # Logging step, after each log_step batches a log will be recorded
        :param checkpoints_dir_path:  # Checkpoints directory to save the model training progress and checkpoints
        """
        self.model = model
        self.dataloader = dataloader
        self.cls_loss = cls_loss_function()
        self.adv_loss = adv_loss_function()
        self.jigsaw_loss = jigsaw_loss_function()
        self.use_adv = use_adv
        self.use_jigsaw = use_jigsaw
        self.optimizer = optimizer
        self.epochs = epochs
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.log_step = log_step
        self.checkpoints_dir_path = checkpoints_dir_path
        self.validator = DCLTester(test_dataloader, cls_loss_function) if test_dataloader else None
        self.metrics = {}

    def train_epoch(self, epoch):
        """
        The function trains the model for one epoch.
        """
        total_loss = 0
        total_predictions = 0
        total_correct_predictions = 0
        self.model.train()
        for batch_idx, d in enumerate(self.dataloader):
            inputs, labels, labels_jigsaw, patch_labels = d
            inputs = inputs.to(self.device)
            labels = Variable(torch.from_numpy(np.array(labels))).to(self.device)
            labels_jigsaw = Variable(torch.from_numpy(np.array(labels_jigsaw))).to(self.device)
            patch_labels = Variable(torch.from_numpy(np.array(patch_labels))).float().to(self.device)
            cls_outputs, adv_outputs, jigsaw_mask_outputs = self.model(inputs, train=True)
            cls_loss = self.cls_loss(cls_outputs, labels)
            adv_loss = self.adv_loss(adv_outputs, labels_jigsaw)
            # jigsaw reconstruct uses regression type with l1  or mse loss or class with bce loss
            jigsaw_loss = self.jigsaw_loss(jigsaw_mask_outputs, patch_labels)
            loss = cls_loss
            if self.use_adv:
                loss += adv_loss
            if self.use_jigsaw:
                loss += jigsaw_loss
            total_loss += loss
            _, preds = torch.max(cls_outputs, 1)
            total_predictions += len(preds)
            total_correct_predictions += torch.sum(preds == labels.data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if (batch_idx % self.log_step == 0) and (batch_idx != 0):
                logger.info(
                    f"Train Epoch: {epoch}, Step, {batch_idx}/{len(self.dataloader)}, Loss: {total_loss / batch_idx}")
        self.metrics[epoch] = {}
        self.metrics[epoch]["train"] = {}
        self.metrics[epoch]["train"]["loss"] = float(total_loss / batch_idx)
        self.metrics[epoch]["train"]["accuracy"] = float(total_correct_predictions) / float(total_predictions)
        logger.info(f"Epoch {epoch} loss: {self.metrics[epoch]['train']['loss']}, accuracy:, "
                    f"{self.metrics[epoch]['train']['accuracy']}")

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
