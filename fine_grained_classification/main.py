import sys
import os

# Add the root folder (Visitor Tracking Utils) as the path to modules.
sys.path.append(f"{'/'.join(os.getcwd().split('/')[:-1])}")

from fine_grained_classification.config.config import Configuration as config
from fine_grained_classification.dataloader.common import Dataloader
from fine_grained_classification.model.common import Model
from fine_grained_classification.train.common import Trainer

if __name__ == "__main__":
    config.load_config("./config.yml")
    # Create the dataloaders
    dataloader = Dataloader(config=config)
    train_loader, test_loader = dataloader.get_loader()
    # Create the model
    model = Model(config=config).get_model()
    # Create the trainer
    trainer = Trainer(config=config, model=model, dataloader=train_loader, val_dataloader=test_loader).get_trainer()
    trainer.train_and_validate()

    # iterations = 0
    # with torch.no_grad():
    #     for train_sample, test_sample in zip(train_loader, test_loader):
    #         images, label = train_sample
    #         images = images.to("cuda")
    #         label = label.to("cuda")
    #         start = time.time()
    #         logits = model(images)
    #         print(f"Batch Inference Time: {(time.time() - start) * 1000} ms")
    #         iterations += 1
    #         if iterations == 10:
    #             break
    # print("done")
