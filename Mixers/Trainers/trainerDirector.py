from cgitb import text
from .classificationTrainer import ClassificationTrainer, ClassificationTrainerHMC

from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset

from torchmetrics import ConfusionMatrix, Accuracy, Recall, Precision


class TrainerDirector():
    
    @staticmethod
    def get_binary_trainer(model:nn.Module, device, traindataset:Dataset, testdataset:Dataset, nb_epochs:int=20, batch_size:int=256, collate_fn=None) -> ClassificationTrainer:
        trainer = ClassificationTrainer(model, nb_epochs=nb_epochs, device=device, traindataset=traindataset, testdataset=testdataset, batch_size=batch_size, collate_fn=collate_fn)
        return trainer.set_optimizer_and_loss(nn.BCELoss, optim.Adam).set_validation_metrics({ConfusionMatrix, Accuracy, Recall, Precision}, num_classes=2)

    @staticmethod
    def get_multiclass_trainer(model:nn.Module, device, traindataset:Dataset, testdataset:Dataset, num_classes:int, nb_epochs:int=20, batch_size:int=256, collate_fn=None) -> ClassificationTrainer:
        trainer = ClassificationTrainer(model, nb_epochs=nb_epochs, device=device, traindataset=traindataset, testdataset=testdataset, batch_size=batch_size, collate_fn=collate_fn)
        return trainer.set_optimizer_and_loss(nn.CrossEntropyLoss, optim.Adam).set_validation_metrics({ConfusionMatrix, Accuracy}, num_classes=num_classes)
        
    @staticmethod
    def get_hmc_trainer(model:nn.Module, device, traindataset:Dataset, testdataset:Dataset, batch_size:int=256, collate_fn=None) -> ClassificationTrainer:
        trainer = ClassificationTrainerHMC(model, device=device, traindataset=traindataset, testdataset=testdataset, batch_size=batch_size, collate_fn=collate_fn)
        return trainer.set_loss(nn.CrossEntropyLoss).set_validation_metrics({ConfusionMatrix, Accuracy, Recall, Precision}, num_classes=2)
