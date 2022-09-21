from Mixers.Datasets.DSP import IMDBSentimentAnalysis
from Mixers.NLPMixer.nlpmixer import NLP_Mixer
from Mixers.Trainers.trainerDirector import TrainerDirector

import torch

if __name__ == "__main__":
    useGPU = True
    
    traindataset = IMDBSentimentAnalysis(limit=1000)
    testdataset = IMDBSentimentAnalysis(train=False)
    model = NLP_Mixer(sentenceLength=200, depth=2, applyPreprocessing=False)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() and useGPU else "cpu")
    
    trainer = TrainerDirector.get_hmc_trainer(model=model, traindataset=traindataset, testdataset=testdataset, device=device) 
    
    trainer.summarize_model()
    
    # trainer.load_model("saves/NLP_Mixer_Binary-2022-09-12-12:44")
    
    trainer.train()
    
    # trainer.save_model()
    
    trainer.validate(light=True)

