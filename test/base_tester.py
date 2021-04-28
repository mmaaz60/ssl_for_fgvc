import torch
import logging


logger = logging.getLogger(f"test/base_tester.py")


class BaseTester:
    def __init__(self, dataloader, loss_function, device="cuda"):
        self.dataloader = dataloader
        self.loss = loss_function()
        self.device = device

    def test(self, model):
        metrics = {}
        model.eval()
        with torch.no_grad():
            total_loss = 0
            total_correct_predictions_top_1 = 0
            total_correct_predictions_top_2 = 0
            for batch_idx, d in enumerate(self.dataloader):
                inputs, labels = d
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = model(inputs, train=False)
                loss = self.loss(outputs, labels)
                total_loss += loss
                top_2_val, top_2_pos = torch.topk(outputs, 2)
                batch_corrects_1 = torch.sum((top_2_pos[:, 0] == labels)).data.item()
                total_correct_predictions_top_1 += batch_corrects_1
                batch_corrects_2 = torch.sum((top_2_pos[:, 1] == labels)).data.item()
                total_correct_predictions_top_2 += (batch_corrects_2 + batch_corrects_1)
            metrics['loss'] = float(total_loss) / len(self.dataloader)
            metrics['accuracy_top_1'] = float(total_correct_predictions_top_1) / (
                                        len(self.dataloader) * self.dataloader.batch_size)
            metrics['accuracy_top_2'] = float(total_correct_predictions_top_2) / (
                                        len(self.dataloader) * self.dataloader.batch_size)
            logger.info(f"Validation loss: {metrics['loss']}, Top-1 Validation accuracy: {metrics['accuracy_top_1']}"
                        f", Top-2 Validation accuracy: {metrics['accuracy_top_2']}")
        return metrics
