from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import Dataset

from mixers.utils.helper import InteractivePlot, generate_dashboard
import mixers.trainers.hamiltorch as hamiltorch
from mixers.trainers.abstractTrainers import RegressionTrainerAbstract

from rich.align import Align
from rich.panel import Panel
from rich.progress import Progress

class RegressionTrainerHMC(RegressionTrainerAbstract):

    def __init__(self, model:nn.Module, device='cpu', save_path:str="saves/", num_samples:int=100,
                 traindataset:Dataset=None, testdataset:Dataset=None, evaldataset:Dataset=None, 
                 batch_size:int=256, collate_fn=None, **kwargs) -> None:
        super().__init__(model, device, save_path, traindataset=traindataset, testdataset=testdataset, evaldataset=evaldataset, 
                        batch_size=batch_size, collate_fn=collate_fn, kwargs=kwargs)
        
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
        super().train()        

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

    
    # Helper functions =========================================================================================================== 
    def save_model(self, savePath="saves"):
        object_to_save = self.params_hmc_f
        return super().save_model(object_to_save, savePath=savePath)
    
    def load_model(self, load_path):
        self.params_hmc_f = torch.load(load_path)
        print(self.params_hmc_f)