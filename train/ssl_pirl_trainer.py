import torch
from test.base_tester import BaseTester
from utils.util import save_model_checkpoints
import logging

logger = logging.getLogger(f"train/ssl_pirl_trainer.py")


class SSLPIRLTrainer:
    def __init__(self, model, dataloader, loss_function, optimizer, epochs, memory,
                 lr_scheduler=None, val_dataloader=None, device="cuda", log_step=50, checkpoints_dir_path=None):
        """
        Constructor, the function initializes the training related parameters.

        :param model: The model to train
        :param dataloader: The dataloader to get training samples from
        :param loss_function:  The loss function
        :param optimizer: The optimizer to be used for training
        :param epochs: Number of epochs
        :param memory: The instance of memory bank to be used for PIRL
        :param lr_scheduler:  Learning rate scheduler
        :param val_dataloader: Validation dataloader to get the validation samples
        :param device:  The execution device
        :param log_step:  # Logging step, after each log_step batches a log will be recorded
        :param checkpoints_dir_path:  # Checkpoints directory to save the model training progress and checkpoints
        """
        self.model = model
        self.dataloader = dataloader
        self.loss = loss_function()
        self.optimizer = optimizer
        self.epochs = epochs
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.log_step = log_step
        self.checkpoints_dir_path = checkpoints_dir_path
        self.validator = BaseTester(val_dataloader, loss_function) if val_dataloader else None
        self.memory = memory.to(self.device)
        self.metrics = {}

    @staticmethod
    def _compute_pirl_loss(logits, target, criterion):
        """
        The function computes the PIRL losses. The implementation is based on https://github.com/HobbitLong/PyContrast

        :param logits: a list of logits, each with a contrastive task
        :param target: contrastive learning target
        :param criterion: typically nn.CrossEntropyLoss
        """
        losses = [criterion(logit, target) for logit in logits]

        return losses

    def train_epoch(self, epoch):
        """
        The function trains the model for one epoch.
        """
        total_loss_cls = 0
        total_loss_pirl = 0
        total_loss = 0
        total_predictions = 0
        total_correct_predictions = 0
        self.model.train()
        for batch_idx, d in enumerate(self.dataloader):
            o, _, x_jig, labels, index = d  # Parse the inputs
            # Transfer the data to GPU
            o = o.to(self.device)
            x_jig = x_jig.to(self.device)
            bsz, m, c, h, w = x_jig.shape
            x_jig = x_jig.view(bsz * m, c, h, w)
            labels = labels.to(self.device)
            index = index.to(self.device)
            # Generate predictions
            classification_scores, representation, representation_jig = self.model(o, x_jig, train=True)
            pirl_output = self.memory(representation, index, representation_jig)
            # Compute loss
            cls_loss = self.loss(classification_scores, labels)
            pirl_losses = self._compute_pirl_loss(logits=pirl_output[:-1], target=pirl_output[-1], criterion=self.loss)
            pirl_loss = (1 - 0.5) * pirl_losses[0] + 0.5 * pirl_losses[1]
            loss = cls_loss + pirl_loss
            total_loss_cls += cls_loss
            total_loss_pirl += pirl_loss
            total_loss += loss
            # Calculate matrix
            _, preds = torch.max(classification_scores, 1)
            total_predictions += len(preds)
            total_correct_predictions += torch.sum(preds == labels.data)
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if (batch_idx % self.log_step == 0) and (batch_idx != 0):
                logger.info(
                    f"Train Epoch: {epoch}, Step, {batch_idx}/{len(self.dataloader)}, "
                    f"Cls Loss: {total_loss_cls / batch_idx}, PIRL Loss: {total_loss_pirl / batch_idx}, "
                    f"Combined Loss: {total_loss / batch_idx}")
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
