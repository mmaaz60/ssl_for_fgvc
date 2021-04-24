import torch
from test.base_tester import BaseTester
import logging

logger = logging.getLogger(f"train/ssl_pirl_trainer.py")


class SSLPIRLTrainer:
    def __init__(self, model, dataloader, loss_function, pirl_loss_weight, optimizer, epochs, memory,
                 lr_scheduler=None, val_dataloader=None, device="cuda", log_step=50, checkpoints_dir_path=None):
        self.model = model
        self.dataloader = dataloader
        self.loss = loss_function()
        self.pirl_loss_weight = pirl_loss_weight
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
        Args:
          logits: a list of logits, each with a contrastive task
          target: contrastive learning target
          criterion: typically nn.CrossEntropyLoss
        """
        losses = [criterion(logit, target) for logit in logits]

        return losses

    def train_epoch(self, epoch):
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
            loss = (1 - self.pirl_loss_weight) * cls_loss + self.pirl_loss_weight * pirl_loss
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
                    f"Cls Loss: {total_loss_cls / batch_idx}, PIRL Loss: {total_loss_pirl / batch_idx}"
                    f"Combined Loss: {total_loss / batch_idx}")
        self.metrics[epoch] = {}
        self.metrics[epoch]["train"] = {}
        self.metrics[epoch]["train"]["loss"] = float(total_loss / batch_idx)
        self.metrics[epoch]["train"]["accuracy"] = float(total_correct_predictions) / float(total_predictions)
        logger.info(f"Epoch {epoch} loss: {self.metrics[epoch]['train']['loss']}, accuracy:, "
                    f"{self.metrics[epoch]['train']['accuracy']}")

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
