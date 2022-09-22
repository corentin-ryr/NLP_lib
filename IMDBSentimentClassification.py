from Mixers.Datasets.DSP import IMDBSentimentAnalysis
from Mixers.NLPMixer.nlpmixer import NLP_Mixer
from Mixers.Trainers.trainerDirector import TrainerDirector
from Mixers.Helper.helper import get_device

import torch

if __name__ == "__main__":
    useGPU = False
    device = get_device(useGPU)
    
    traindataset = IMDBSentimentAnalysis()
    testdataset = IMDBSentimentAnalysis(train=False)
    model = NLP_Mixer(sentenceLength=100, depth=2, device=device)
    
    trainer = TrainerDirector.get_binary_trainer(model=model, traindataset=traindataset, testdataset=testdataset, batch_size=256, device=device, nb_epochs=20) 
    
    trainer.summarize_model()
    
    # trainer.load_model("saves/NLP_Mixer_Binary-2022-09-12-12:44")
    
    trainer.train()
    
    trainer.save_model()
    
        
    trainer.validate()

