import sys
import os
import argparse
import torch
from torch.autograd import Variable
import numpy as np

# Add the root folder (ssl_for_fgvc) as the path
sys.path.append(f"{'/'.join(os.getcwd().split('/')[:-1])}")
from config.config import Configuration as config
from dataloader.common import Dataloader
from model.common import Model
from utils.util import get_object_from_path


def parse_arguments():
    """
    Parse the command line arguments
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-config", "--config_path", required=True,
                    help="The path to the pipeline .yml configuration file.")
    ap.add_argument("-checkpoints", "--model_checkpoints", required=True,
                    help="The path to model checkpoints.")
    ap.add_argument("-dataset", "--root_dataset_path", required=False, default="./data/CUB_200_2011",
                    help="The path to the dataset root directory. "
                         "The program will download the dataset if not present locally.")
    ap.add_argument("-d", "--device", required=False, default='cuda',
                    help="The computation device to perform operations ('cpu', 'cuda')")

    args = vars(ap.parse_args())

    return args


def main():
    """
    Implements the main flow, i.e. load the dataset & model, generate cam visualizations and save the visualizations
    """
    args = parse_arguments()  # Parse arguments
    config.load_config(args["config_path"])  # Load configuration
    config.cfg["dataloader"]["root_directory_path"] = args["root_dataset_path"]  # Set the dataset path
    _, test_loader = Dataloader(config=config).get_loader()  # Create dataloader
    # Create the model
    model = Model(config=config).get_model()
    model = model.to(args["device"])
    # Load pretrained weights
    checkpoints_path = args["model_checkpoints"]
    checkpoints = torch.load(checkpoints_path)
    model.load_state_dict(checkpoints["state_dict"], strict=True)
    # Initialize the loss
    loss_func = get_object_from_path(config.cfg["train"]["class_loss_function_path"])
    test_loss = loss_func()
    # Iterate over the test dataset and calculate the metrics
    total_loss = 0  # Variable to store the running loss
    total_correct_predictions_top_1 = 0  # Variable to store the total top-1 correction predictions
    total_correct_predictions_top_2 = 0  # Variable to store the total top-1 correction predictions
    total_predictions = 0  # Variable to store the total predictions
    metrics = {}  # Dictionary to store the evaluation metrics
    model.eval()  # Put the model in the evaluation mode
    with torch.no_grad():  # Require to continuously free the GPU memory after inference
        print(f"Evaluating. It may take some time. Thank you for your patience.")
        for batch_idx, d in enumerate(test_loader):
            inputs, labels = d  # Extract inputs and labels
            # Move the data on the specified device
            inputs = inputs.to(args["device"])
            try:
                labels = labels.to(args["device"])
            except Exception:
                labels = Variable(torch.from_numpy(np.array(labels))).long().to(args["device"])
            total_predictions += len(labels)  # Increment total predictions
            outputs = model(inputs, train=False)  # Perform batch inference
            loss = test_loss(outputs, labels)  # Calculate the loss
            total_loss += loss  # Total loss till now
            top_2_val, top_2_pos = torch.topk(outputs, 2)  # Top 2 values and corresponding positions
            # Calculate the top-1 and top-2 accuracies
            batch_corrects_1 = torch.sum((top_2_pos[:, 0] == labels)).data.item()
            total_correct_predictions_top_1 += batch_corrects_1
            batch_corrects_2 = torch.sum((top_2_pos[:, 1] == labels)).data.item()
            total_correct_predictions_top_2 += (batch_corrects_2 + batch_corrects_1)
            if batch_idx % 50 == 0:
                metrics['loss'] = round(float(total_loss) / len(test_loader), 4)
                metrics['accuracy_top_1'] = round(float(total_correct_predictions_top_1) / total_predictions, 4)
                metrics['accuracy_top_2'] = round(float(total_correct_predictions_top_2) / total_predictions, 4)
                print(f"Step {batch_idx}/{len(test_loader)}, Test loss: {metrics['loss']}, "
                      f"Top-1 Test accuracy: {metrics['accuracy_top_1']}"
                      f", Top-2 Test accuracy: {metrics['accuracy_top_2']}")
        # Final Scores
        print(f"Final Scores.")
        metrics['loss'] = float(total_loss) / len(test_loader)
        metrics['accuracy_top_1'] = float(total_correct_predictions_top_1) / total_predictions
        metrics['accuracy_top_2'] = float(total_correct_predictions_top_2) / total_predictions
        print(f"Test loss: {metrics['loss']}, Top-1 Test accuracy: {metrics['accuracy_top_1']}"
              f", Top-2 Test accuracy: {metrics['accuracy_top_2']}")


if __name__ == "__main__":
    main()
