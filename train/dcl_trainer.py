import torch
from test.base_tester import BaseTester
import logging

logger = logging.getLogger(f"train/base_trainer.py")


class DCLTrainer:
    def __init__(self, model, dataloader, cls_loss_function, adv_loss_function, jigsaw_loss_function, optimizer, epochs,
                 lr_scheduler=None, test_dataloader=None, device="cuda", log_step=50, checkpoints_dir_path=None):
        self.model = model
        self.dataloader = dataloader
        self.cls_loss = cls_loss_function()
        self.adv_loss = adv_loss_function()
        self.jigsaw_loss = jigsaw_loss_function()
        self.optimizer = optimizer
        self.epochs = epochs
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.log_step = log_step
        self.checkpoints_dir_path = checkpoints_dir_path
        self.validator = BaseTester(test_dataloader, cls_loss_function) if test_dataloader else None
        self.metrics = {}

    def train_epoch(self, epoch):
        total_loss = 0
        total_predictions = 0
        total_correct_predictions = 0
        self.model.train()
        for batch_idx, d in enumerate(self.dataloader):
            inputs, labels, labels_jigsaw, patch_labels = d
            inputs = inputs.to(self.device)
            labels, labels_jigsaw = labels.to(self.device), labels_jigsaw.to(self.device)
            patch_labels = patch_labels.to(self.device)
            cls_outputs, adv_outputs, jigsaw_mask_outputs = self.model(inputs)
            cls_loss = self.cls_loss(cls_outputs, labels)
            adv_loss = self.adv_loss(adv_outputs, labels_jigsaw)
            jigsaw_loss = self.jigsaw_loss(jigsaw_mask_outputs, patch_labels)
            loss = cls_loss + adv_loss + jigsaw_loss
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
