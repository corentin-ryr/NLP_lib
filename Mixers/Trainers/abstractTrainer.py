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




class Trainer(ABC):
    def __init__(self, model:nn.Module, device, save_path:str, traindataset:Dataset=None, testdataset:Dataset=None, evaldataset:Dataset=None, batch_size:int=256, collate_fn=None, **kwargs) -> None:
        self.model:nn.Module = model.to(device)
        self.device = device
        self.save_path = save_path

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
        
    def save_model(self, object_to_save=None):
        path = os.path.join(self.save_path, self.model._get_name() + "-" + datetime.today().strftime('%Y-%m-%d-%H-%M'))
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