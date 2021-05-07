import torch
import logging
from torch.autograd import Variable
import numpy as np

logger = logging.getLogger(f"test/ssl_dcl_tester.py")


class DCLTester:
    """
    The class implements the dcl tester for the training pipeline.
    """
    def __init__(self, dataloader, loss_function, device="cuda"):
        """
        Constructor, the function initializes the required parameters.

        :param dataloader: The test dataloader
        :param loss_function: The loss function for calculating the loss
        :param device: Device of execution
        """
        self.dataloader = dataloader
        self.loss = loss_function()
        self.device = device

    def test(self, model):
        """
        The function implements the testing pipeline.

        :param model: The model to be used for the testing
        """
        metrics = {}  # Dictionary to store the evaluation metrics
        model.eval()  # Put the model in the evaluation mode
        with torch.no_grad():  # Require to continuously free the GPU memory after inference
            total_loss = 0  # Variable to store the running loss
            total_correct_predictions_top_1 = 0  # Variable to store the total top-1 correction predictions
            total_correct_predictions_top_2 = 0  # Variable to store the total top-1 correction predictions
            total_predictions = 0  # Variable to store the total predictions
            # Iterate over the dataset
            for batch_idx, d in enumerate(self.dataloader):
                inputs, labels = d  # Extract inputs and labels
                # Move the data on the specified device
                inputs = inputs.to(self.device)
                labels = Variable(torch.from_numpy(np.array(labels))).long().to(self.device)
                total_predictions += len(labels)  # Increment total predictions
                outputs = model(inputs, train=False)  # Perform batch inference
                loss = self.loss(outputs, labels)  # Calculate the loss
                total_loss += loss  # Total loss till now
                top_2_val, top_2_pos = torch.topk(outputs, 2)   # Top 2 values and corresponding positions
                # Calculate the top-1 and top-2 accuracies
                batch_corrects_1 = torch.sum((top_2_pos[:, 0] == labels)).data.item()
                total_correct_predictions_top_1 += batch_corrects_1
                batch_corrects_2 = torch.sum((top_2_pos[:, 1] == labels)).data.item()
                total_correct_predictions_top_2 += (batch_corrects_2 + batch_corrects_1)
            metrics['loss'] = float(total_loss) / len(self.dataloader)
            metrics['accuracy_top_1'] = float(total_correct_predictions_top_1) / total_predictions
            metrics['accuracy_top_2'] = float(total_correct_predictions_top_2) / total_predictions
            logger.info(f"Validation loss: {metrics['loss']}, Top-1 Validation accuracy: {metrics['accuracy_top_1']}"
                        f", Top-2 Validation accuracy: {metrics['accuracy_top_2']}")
        return metrics
