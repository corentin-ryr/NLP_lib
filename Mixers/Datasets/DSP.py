import os
import numpy as np

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

class IMDBSentimentAnalysis(Dataset):
    
    def __init__(self, train=True, shuffle=True, limit=None) -> None:
        
        self.dirpath = "data/imdb/" + ("train" if train else "test")
        
        fileListPos = os.listdir(os.path.join(self._get_path(), "pos"))
        fileListNeg = os.listdir(os.path.join(self._get_path(), "neg"))
        
        self.samples = np.concatenate(( 
                                       np.array(fileListNeg), 
                                       np.array(fileListPos) 
                                       ))
        self.labels = torch.cat(( 
                                      torch.zeros((len(fileListNeg)), dtype=torch.long), 
                                      torch.ones((len(fileListPos)), dtype=torch.long) 
                                      ))
        
        if shuffle: self._shuffle()
        if limit: self.limit = limit
        else: self.limit =  float("inf")
    
    def _shuffle(self):
        shuffleIndex = np.arange(len(self.labels))
        np.random.shuffle(shuffleIndex)
        
        self.samples = self.samples[shuffleIndex]
        self.labels = self.labels[shuffleIndex]
        
    def _read_file(self, idx):
        filename = self.samples[idx]
        label = "pos" if self.labels[idx] else "neg"
        
        with open(os.path.join(self.dirpath, label, filename), 'r') as file:
            return "".join(file.readlines())
        
    def _get_path(self):
        return os.path.join(os.getcwd(), self.dirpath)
    
    def __len__(self):
        return min([len(self.labels), self.limit])
    
    def __getitem__(self, idx):
        
        label = self.labels[idx]
        sample = self._read_file(idx)    
        
        return sample, label
       
       
class MTOPEnglish(Dataset):
    
    sets = {"train", "test", "eval"}
    labelMap = {"alarm": 0,
                "calling": 1,
                "event": 2,
                "messaging": 3,
                "music": 4,
                "news": 5,
                "people": 6,
                "recipes": 7,
                "reminder": 8,
                "timer": 9,
                "weather": 10}
    
    def __init__(self, set:str="train", limit:int=None) -> None:
        super().__init__() 
        
        if not set in MTOPEnglish.sets: raise ValueError("Invalid value for set.") 
        
        self.path = os.path.join("data/mtop/en", set + ".txt")
        self.limit = limit if limit else float("inf")
        self.samples = []
        self.labels = []
        
        
        with open(self.path, 'r') as file:
            lines = file.readlines()
            self.length = len(lines)
            
            for line in lines:
                line = line.split("\t")
                self.samples.append(line[3])
                self.labels.append(MTOPEnglish.labelMap[line[4]])
        
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        

        
    def __len__(self):
        return min([self.length, self.limit])
    
    def __getitem__(self, idx):
        
        label = self.labels[idx]
        sample = self.samples[idx]
        
        return sample, label




if __name__ == "__main__":

    dataset = IMDBSentimentAnalysis()
    dataset = MTOPEnglish()
    
    dataloader = DataLoader(dataset, batch_size=5)
    for sample, label in tqdm(dataloader, desc="Reading train samples."):
        # print(sample)
        # print(label)
        pass

    print("All samples read.")