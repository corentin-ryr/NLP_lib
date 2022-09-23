from Mixers.Datasets.DSP import IMDBSentimentAnalysis
from Mixers.NLPMixer.nlpmixer import NLP_Mixer
from Mixers.Trainers.trainerDirector import TrainerDirector
from Mixers.Helper.helper import get_device

import torch


sentenceLength = 100
textFormat = "raw"

if __name__ == "__main__":
    useGPU = False
    device = get_device(useGPU)

    traindataset = IMDBSentimentAnalysis(textFormat=textFormat, sentenceLength=sentenceLength)
    testdataset = IMDBSentimentAnalysis(train=False, textFormat=textFormat, sentenceLength=sentenceLength)

    model = NLP_Mixer(sentenceLength=200, depth=2, applyPreprocessing=False)
    
    trainer = TrainerDirector.get_hmc_trainer(model=model, traindataset=traindataset, testdataset=testdataset, device=device) 
    
    trainer.summarize_model()
    
    # trainer.load_model("saves/NLP_Mixer_Binary-2022-09-12-12:44")
    
    trainer.train()
    
    # trainer.save_model()
    
    trainer.validate(light=True)

