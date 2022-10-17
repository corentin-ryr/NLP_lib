from abc import ABC, abstractmethod
import os
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from rich.panel import Panel
from rich.pretty import Pretty
from rich.console import Console
from rich.align import Align

from torchmetrics.metric import Metric
from torchmetrics import ConfusionMatrix, Accuracy, Recall, Precision, MeanSquaredError



class Trainer(ABC):
    def __init__(self, model:nn.Module, device, traindataset:Dataset=None, testdataset:Dataset=None, evaldataset:Dataset=None, batch_size:int=256, collate_fn=None, **kwargs) -> None:
        self.model:nn.Module = model.to(device)
        self.device = device

        if "shuffle" in kwargs:
            shuffle = kwargs["shuffle"]
        else: shuffle = True

        if traindataset: 
            self.trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
            if collate_fn: self.trainloader.collate_fn = collate_fn
        if testdataset: 
            self.testloader = DataLoader(testdataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
            if collate_fn: self.testloader.collate_fn = collate_fn
        if evaldataset: 
            self.evalloader = DataLoader(evaldataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
            if collate_fn: self.evalloader.collate_fn = collate_fn

        self.batch_size = batch_size
        
        self.console = Console()

        
    def summarize_model(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        prettyModel = Pretty(self.model)
        self.console.print(Panel(prettyModel, title=f"[green]Device {self.device} | Number of parameters: {total_params}", border_style="green"))
        
    def save_model(self, object_to_save=None, savePath="saves"):
        path = os.path.join(savePath, self.model._get_name() + "-" + datetime.today().strftime('%Y-%m-%d-%H-%M'))
        if not os.path.exists(savePath):
            os.makedirs(savePath)

        if not object_to_save:
            torch.save(self.model.state_dict(), path)
        else:
            torch.save(object_to_save, path)
    
    def load_model(self, load_path):
        self.model.load_state_dict(torch.load(load_path))

    @abstractmethod
    def train(self):
        self.console.print(Align("\n\nStarting training at " + datetime.now().strftime("%H:%M:%S"), align="center"))


    @abstractmethod
    def validate(self):
        pass


class ClassificationTrainerAbstract(Trainer):

    def __init__(self, model: nn.Module, device, traindataset: Dataset = None, testdataset: Dataset = None, evaldataset: Dataset = None, batch_size: int = 256, collate_fn=None, num_classes=2, **kwargs) -> None:
        super().__init__(model, device, traindataset, testdataset, evaldataset, batch_size, collate_fn, **kwargs)
        

        if "loss" in kwargs: self.criterion = kwargs["loss"]
        else:
            if num_classes == 2: self.criterion = nn.BCELoss
            else: self.criterion = nn.CrossEntropyLoss


        self.num_classes = num_classes
        self.metric_set:set[Metric] = set()
        if "metric_set" in kwargs:
            self.metric_set = kwargs["metric_set"]
        else:
            if num_classes == 2: self.metric_set = {ConfusionMatrix, Accuracy, Recall, Precision}
            else: self.metric_set = {ConfusionMatrix, Accuracy}

        self._init_metrics_loss()

    # Helper functions ===========================================================================================================  
    supportedMetrics = {Accuracy, Recall, Precision}
    def _init_metrics_loss(self):
        new_metric_set = set()
        for metric in self.metric_set:
            if metric in ClassificationTrainerAbstract.supportedMetrics: 
                new_metric_set.add(metric(self.num_classes) if self.num_classes > 2 else metric())
            if metric in {ConfusionMatrix}:
                new_metric_set.add(metric(self.num_classes))
        self.metric_set = new_metric_set
        
        self.criterion = self.criterion()


class RegressionTrainerAbstract(Trainer):

    def __init__(self, model: nn.Module, device, traindataset: Dataset = None, testdataset: Dataset = None, evaldataset: Dataset = None, batch_size: int = 256, collate_fn=None, **kwargs) -> None:
        super().__init__(model, device, traindataset, testdataset, evaldataset, batch_size, collate_fn, **kwargs)
        
        if "loss" in kwargs: self.criterion = kwargs["loss"]
        else:
            self.criterion = nn.MSELoss

        self.metric_set:set[Metric] = set()
        if "metric_set" in kwargs:
            self.metric_set = kwargs["metric_set"]
        else:
            self.metric_set = {MeanSquaredError}

        self._init_metrics_loss()

    # Helper functions ===========================================================================================================  
    supportedMetrics = {MeanSquaredError}
    def _init_metrics_loss(self):
        new_metric_set = set()
        for metric in self.metric_set:
            if metric in ClassificationTrainerAbstract.supportedMetrics: 
                new_metric_set.add(metric())
        self.metric_set = new_metric_set
        
        self.criterion = self.criterion()
