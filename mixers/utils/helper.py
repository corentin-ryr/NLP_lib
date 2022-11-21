from typing import List
import matplotlib.pyplot as plt
import random

from rich.table import Table
from rich.layout import Layout

import torch

from torchmetrics import ConfusionMatrix
from torchmetrics.metric import Metric


class InteractivePlot():
    
    def __init__(self, num_axes:int, axes_names=None) -> None:
        """
        num_axes is the number of variables that should be plotted.
        """
        self.i = []
        self.val = []
        plt.ion()
        self.figure, self.ax1 = plt.subplots()
        self.ax1.set_title("Loss every epoch")

        
        self.val.append([])
        self.axes = [self.ax1]
        self.lines =[self.ax1.plot([], self.val[0], '-', c=[random.random() for _ in range(3)], linewidth=1.5, markersize=4, label=axes_names[0] if axes_names else "")[0]]
        

        for i in range(1, num_axes):
            ax = self.ax1.twinx()

            self.val.append([])
            self.axes.append(ax)
            self.lines.append(ax.plot([], self.val[0], '-', c=[random.random() for _ in range(3)], linewidth=1.5, markersize=4, label=axes_names[i] if axes_names else "")[0])
        
        self.ax1.legend(handles=self.lines)


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

            self.axes[index].set_xlim(0, len(self.val[0]))
            self.axes[index].set_ylim(0, max(self.val[index]) * 1.5)

        plt.draw()
        plt.pause(1e-20)

    def save_plot(self, path):
        self.figure.savefig(path + "/loss.png")   
    
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
