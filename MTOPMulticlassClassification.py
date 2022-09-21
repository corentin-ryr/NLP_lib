from Mixers.Datasets.DSP import MTOPEnglish
from Mixers.NLPMixer.nlpmixer import NLP_Mixer
from Mixers.Trainers.trainerDirector import TrainerDirector

import torch

if __name__ == "__main__":
    useGPU = True
    device = torch.device("cuda:0" if torch.cuda.is_available() and useGPU else "cpu")
    
    traindataset = MTOPEnglish()
    testdataset = MTOPEnglish(set="test")
    model = NLP_Mixer(sentenceLength=20, depth=2, nbClasses=11, device=device)
    
    
    trainer = TrainerDirector.get_multiclass_trainer(model=model, traindataset=traindataset, testdataset=testdataset, num_classes=11, batch_size=256, device=device, nb_epochs=20) 
    
    trainer.summarize_model()
    
    # trainer.load_model("saves/NLP_Mixer_Binary-2022-09-12-12:44")
    
    trainer.train()
    
    trainer.save_model()
    
        
    trainer.validate()

