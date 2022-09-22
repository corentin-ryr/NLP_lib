from typing import Union
import os
import numpy as np

import tkinter as tk
from tkinter import filedialog

import torch
from torch import nn

from rich.prompt import Prompt

class InteractiveInferenceClassification():

    def __init__(self, model:nn.Module, loadFromFile:Union[str, bool]=True, device:str="cpu", classesName:list[str]=None) -> None:
        self.model = model
        self.loadPath = None
        self.device = device

        self.classesName = classesName

        if type(loadFromFile) == bool:
            self.loadFromFile = loadFromFile
        else:
            self.loadFromFile = True
            self.loadPath = os.path.join(os.getcwd(), loadFromFile)
        

    def start(self):

        if self.loadFromFile and not self.loadPath:
            root = tk.Tk()
            root.withdraw()
            saveDir = os.path.join(os.getcwd(), "saves")

            file_path = filedialog.askopenfilename(title="Select a model save to load", initialdir=saveDir)
            if file_path == "": raise ValueError("No save selected.")
            self.loadPath = file_path
            

        self.model.load_state_dict(torch.load(self.loadPath, map_location=self.device))
        self.model.to(self.device)

        while True:
            input = Prompt.ask("Enter text to infer")
            
            output = self.model([input])
            output = output.cpu().detach().numpy()


            if output.shape == ():
                self.binary_classification(output)
            elif len(output) >= 2:
                self.multiclass_classification(output)


            
        
    def binary_classification(self, output):
        if output < 0.5:
                print("Negative review.")
        else:
            print("Positive review.")

    def multiclass_classification(self, output):
        idx = np.argmax(output)

        if self.classesName and len(self.classesName) == len(output):
            print(self.classesName[idx])
        else:
            print(f"Index of predicted class: {idx}")