from typing import List
import matplotlib.pyplot as plt
import random

from rich.table import Table
from rich.layout import Layout

import torch

from torchmetrics import ConfusionMatrix
from torchmetrics.metric import Metric


class InteractivePlot():
    
    def __init__(self, num_axes:int) -> None:
        """
        num_axes is the number of variables that should be plotted.
        """
        self.i = []
        self.val = []
        plt.ion()
        self.axes = plt.gca()
        self.axes.set_title("Loss every epoch")
        self.lines =[]
        self.vspan = []

        for i in range(num_axes):
            self.val.append([])
            self.lines.append([])
            self.lines[i], = self.axes.plot([], self.val[0], '-', c=[random.random() for _ in range(3)], linewidth=1.5, markersize=4)
            

    def update_plot(self, *args):
        """
        Handles the plotting
        """
        if self.i == []:
            self.i = [0]
        else:
            self.i.append(self.i[-1] + 1)

        for index, val in enumerate(args):
            self.val[index].append(val)
            self.lines[index].set_xdata(self.i)
            self.lines[index].set_ydata(self.val[index])

        self.axes.set_xlim(0, self.nb_epochs)
        self.axes.set_ylim(0, max(max(self.val)) * 1.5, 0.1)
        
        for span in self.vspan:
            span.remove()
        self.vspan.clear()

        plt.draw()
        plt.pause(1e-20)
    
def confusion_matrix_renderable(metric:ConfusionMatrix) -> Table:
    confMat = metric.compute()
                
    confusionTable = Table()
    confusionTable.add_column("")
    for i in range(len(confMat)):
        confusionTable.add_column(f"Label {i}")
    
    for i in range(len(confMat)):
        valList = [str(confMat[i, j].item()) for j in range(len(confMat))]
        confusionTable.add_row(f"Predicted {i}", *valList)
        
    return confusionTable

def generate_dashboard(metric_set:List[Metric]):
    metricsTable = Table()

    metricsTable.add_column("Metric", justify="right", style="cyan", no_wrap=True)
    metricsTable.add_column("Value", style="magenta")

    for metric in metric_set:
        if type(metric) is ConfusionMatrix: 
            confusionTable = confusion_matrix_renderable(metric)

        else:
            metricsTable.add_row(metric._get_name(), '%.3f' % metric.compute().item())
    
    layout = Layout()
    layout.split_row( Layout(metricsTable), Layout(confusionTable) )
    
    return layout

def get_device(useGPU):
    if torch.cuda.is_available() and useGPU:
        return torch.device("cuda:0")
    
    if torch.backends.mps.is_available() and useGPU:
        return torch.device("mps")

    return torch.device("cpu")
