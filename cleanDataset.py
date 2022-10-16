from mixers.datasets.DSP import IMDBSentimentAnalysisDatasetCreator
from torch.utils.data import DataLoader
from tqdm import tqdm

if __name__ == "__main__":

    dataset = IMDBSentimentAnalysisDatasetCreator(train=True)
    
    dataloader = DataLoader(dataset, batch_size=1)
    for sample, label in tqdm(dataloader, desc="Reading train samples"):
        pass
    print(dataset.nbCreated)

    dataset = IMDBSentimentAnalysisDatasetCreator(train=False)
    
    dataloader = DataLoader(dataset, batch_size=1)
    for sample, label in tqdm(dataloader, desc="Reading train samples"):
        pass
    print(dataset.nbCreated)

    print("All samples read.")