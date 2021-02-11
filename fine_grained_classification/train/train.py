import torch


class Train:
    def __init__(self, dataloader, model, loss_function, optimizer, epochs, device="cuda", log_step=50):
        self.dataloader = dataloader
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device
        self.log_step = log_step

    def train_epoch(self, epoch):
        total_loss = 0
        total_predictions = 0
        total_correct_predictions = 0
        self.model.train()
        for batch_idx, d in enumerate(self.dataloader):
            inputs, labels = d
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, labels)
            total_loss += loss
            _, preds = torch.max(outputs, 1)
            total_predictions += len(preds)
            total_correct_predictions += torch.sum(preds == labels.data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if (batch_idx % self.log_step == 0) and (batch_idx != 0):
                print(f"Train Epoch: {epoch}, Loss: {total_loss/batch_idx}")
        epoch_accuracy = float(total_correct_predictions) / float(total_predictions)
        print(f"Epoch {epoch} accuracy: {epoch_accuracy}")

    def train(self):
        for i in range(self.epochs):
            self.train_epoch(i + 1)
