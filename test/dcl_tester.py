import torch
import logging
from torch.autograd import Variable
import numpy as np

logger = logging.getLogger(f"test/ssl_dcl_tester.py")


class DCLTester:
    def __init__(self, dataloader, loss_function, device="cuda"):
        self.dataloader = dataloader
        self.loss = loss_function()
        self.device = device

    def test(self, model):
        metrics = {}
        model.eval()
        with torch.no_grad():
            total_loss = 0
            total_predictions = 0
            total_correct_predictions = 0
            for batch_idx, d in enumerate(self.dataloader):
                inputs, labels = d
                inputs = inputs.to(self.device)
                labels = Variable(torch.from_numpy(np.array(labels))).long().to(self.device)
                cls_outputs = model(inputs, train=False)
                outputs = cls_outputs
                loss = self.loss(outputs, labels)
                total_loss += loss
                _, preds = torch.max(outputs, 1)
                total_predictions += len(preds)
                total_correct_predictions += torch.sum(preds == labels.data)
            metrics['loss'] = float(total_loss) / len(self.dataloader)
            metrics['accuracy'] = float(total_correct_predictions) / total_predictions
            logger.info(f"Validation loss: {metrics['loss']}, Validation accuracy: {metrics['accuracy']}")
        return metrics
