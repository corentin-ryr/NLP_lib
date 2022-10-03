from __future__ import annotations

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

from torchmetrics import ConfusionMatrix, Accuracy, Recall, Precision
from torchmetrics.metric import Metric

from tqdm import tqdm
import os
from datetime import datetime
import math

from Mixers.Helper.helper import InteractivePlot, generate_dashboard
from Mixers.Trainers import hamiltorch

from rich.align import Align
from rich.panel import Panel
from rich.pretty import Pretty
from rich.progress import Progress
from rich.console import Console


class Trainer():
    def __init__(self, model:nn.Module, device, save_path:str, traindataset:Dataset=None, testdataset:Dataset=None, evaldataset:Dataset=None, batch_size:int=256, collate_fn=None) -> None:
        self.model = model.to(device)
        self.device = device
        self.save_path = save_path
        if traindataset: 
            self.trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True, num_workers=4)
            if collate_fn: self.trainloader.collate_fn = collate_fn
        if testdataset: 
            self.testloader = DataLoader(testdataset, batch_size=batch_size, shuffle=True, num_workers=4)
            if collate_fn: self.testloader.collate_fn = collate_fn
        if evaldataset: 
            self.evalloader = DataLoader(evaldataset, batch_size=batch_size, shuffle=True, num_workers=4)
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
    

class ClassificationTrainer(Trainer):

    def __init__(self, model:nn.Module, display_loss:bool=False, nb_epochs:int=20, device='cpu', save_path:str="saves/", 
                 traindataset:Dataset=None, testdataset:Dataset=None, evaldataset:Dataset=None, batch_size:int=256, collate_fn=None) -> None:
        super().__init__(model, device, save_path, traindataset=traindataset, testdataset=testdataset, evaldataset=evaldataset, batch_size=batch_size, collate_fn=collate_fn)
        
        self.display_loss = display_loss
        self.nb_epochs = nb_epochs
        if self.display_loss: self.interactivePlot = InteractivePlot(1)


    def train(self):
        self.console.print(Align("\n\nStarting training", align="center"))

        for epoch in range(self.nb_epochs):  # loop over the dataset multiple times
            self.model.train()

            running_loss = 0.0
            train_accuracy = Accuracy().to(self.device)
            for inputs, labels in tqdm(self.trainloader, total=len(self.trainloader), desc=f"Epoch {epoch}"):

                self.optimizer.zero_grad()
                outputs = self.model(inputs.to(self.device))

                train_accuracy.update(outputs, labels.to(self.device))
                if type(self.criterion) is nn.BCELoss:
                    labels = labels.type(torch.float)
                loss = self.criterion(outputs, labels.to(self.device))
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            
            if self.display_loss: self.interactivePlot.update_plot(running_loss / len(self.trainloader))
            self.console.print(f'Average loss at epoch {epoch}: {running_loss / len(self.trainloader):.3f}')
            self.console.print(f'Acuracy at epoch {epoch}: {train_accuracy.compute().item():.3f}')
            running_loss = 0.0
            
            if epoch % 5 == 4: self.validate(light=True)
        
        self.console.print(Align("\n\n[bold green]Finished Training", align="center"))
        

    def validate(self, light=False):
        self.model.eval()
        for metric in self.metric_set: metric.reset()

        with Progress(transient=True) as progress:
            for i, data in progress.track(enumerate(self.testloader), total=len(self.testloader) if not light else 5):
                if light and i == 5: break
                inputs, labels = data
                outputs = self.model(inputs.to(self.device))
                
                outputs = outputs.detach().cpu()
                labels = labels.detach()
                
                for metric in self.metric_set:
                    metric.update(outputs, labels)
        
        layout = generate_dashboard(self.metric_set)
        self.console.print(Panel(layout, title=f"[green]Validation", border_style="green", height=20))
        
    # Building functions =========================================================================================================== 
    def set_optimizer_and_loss(self, loss:_Loss, optimizer:Optimizer, lr=1e-5) -> ClassificationTrainer:
        self.criterion = loss()
        self.optimizer = optimizer(self.model.parameters(), lr)
        
        return self
        
    supportedMetrics = {Accuracy, Recall, Precision}
    def set_validation_metrics(self, metrics_set:set[Metric], num_classes:int):
        self.num_classes = num_classes
        self.metric_set:set[Metric] = set()
        for metric in metrics_set:
            if metric in ClassificationTrainer.supportedMetrics: 
                self.metric_set.add(metric(num_classes) if num_classes > 2 else metric())
            if metric in {ConfusionMatrix}:
                self.metric_set.add(metric(num_classes))
            
        return self
    

class ClassificationTrainerHMC(Trainer):

    def __init__(self, model:nn.Module, device='cpu', save_path:str="saves/", num_samples:int=100,
                 traindataset:Dataset=None, testdataset:Dataset=None, evaldataset:Dataset=None, batch_size:int=256, collate_fn=None) -> None:
        super().__init__(model, device, save_path, traindataset=traindataset, testdataset=testdataset, evaldataset=evaldataset, batch_size=batch_size, collate_fn=collate_fn)
        
        self.step_size = 0.0005
        self.num_samples = num_samples # 100
        self.tau = 0.1
        self.L = 30 # round(math.pi * self.tau / 2 / self.step_size) # Remember, this is the trajectory length
        self.burn = -1
        self.store_on_GPU = False # This tells sampler whether to store all samples on the GPU
        self.tau_out = 100.
        self.mass = 1.0 # Mass matrix diagonal scale
        
        self.params_hmc_f = []


    def train(self):
        
        self.console.print(Align("\n\nStarting training", align="center"))
        
        params_init = hamiltorch.util.flatten(self.model).to(self.device).clone()
        inv_mass = torch.ones(params_init.shape) / self.mass # Diagonal of inverse mass matrix

        integrator = hamiltorch.Integrator.EXPLICIT
        sampler = hamiltorch.Sampler.HMC # We are doing simple HMC with a standard leapfrog

        self.params_hmc_f = hamiltorch.sample_full_batch_model(self.model, self.trainloader, params_init=params_init,
                                            model_loss="regression", num_samples=self.num_samples,
                                            burn = self.burn, inv_mass=inv_mass.to(self.device), step_size=self.step_size,
                                            num_steps_per_sample=self.L ,tau_out=self.tau_out,
                                            store_on_GPU=self.store_on_GPU, sampler = sampler, integrator=integrator)[1:]

        self.console.print(Align("\n\n[bold green]Finished Sampling", align="center"))
        

    def validate(self, light=False):
        
        with Progress(transient=True) as progress:
            for i, data in progress.track(enumerate(self.testloader), total=len(self.testloader) if not light else 5):
                if light and i == 5: break
                inputs, labels = data
                
                preds = hamiltorch.inference_model(self.model, self.params_hmc_f, inputs)
                outputs = torch.mean(preds, dim=0)
                
                outputs = outputs.detach().cpu()
                labels = labels.detach()

                for metric in self.metric_set:
                    metric.update(outputs, labels)
        
        layout = generate_dashboard(self.metric_set)
        self.console.print(Panel(layout, title=f"[green]Validation", border_style="green", height=20))

    
    # Building functions =========================================================================================================== 
    def save_model(self):
        object_to_save = self.params_hmc_f
        return super().save_model(object_to_save)
    
    def load_model(self, load_path):
        self.params_hmc_f = torch.load(load_path)
        print(self.params_hmc_f)



    
    def set_loss(self, loss:_Loss) -> ClassificationTrainer:
        self.criterion = loss()
        
        return self
        
    supportedMetrics = {Accuracy, Recall, Precision}
    def set_validation_metrics(self, metrics_set:set[Metric], num_classes:int):
        self.num_classes = num_classes
        self.metric_set:set[Metric] = set()
        for metric in metrics_set:
            if metric in ClassificationTrainer.supportedMetrics: 
                self.metric_set.add(metric(num_classes if num_classes > 2 else 1))
            if metric in {ConfusionMatrix}:
                self.metric_set.add(metric(num_classes))
            
        return self

