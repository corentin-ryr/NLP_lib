from Mixers.Datasets.DSP import MTOPEnglish
from Mixers.Utils.preprocessors import collate_callable, ProjectiveLayer
from Mixers.Models.NLPMixer.nlpmixer import NLP_Mixer
from Mixers.Trainers.trainerDirector import TrainerDirector

import torch

sentenceLength = 20
useGPU = True

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() and useGPU else "cpu")
    
    preprocessor = ProjectiveLayer(N=64, S=sentenceLength, M=1024, W=1)

    traindataset = MTOPEnglish()
    testdataset = MTOPEnglish(set="test")

    model = NLP_Mixer(sentenceLength=sentenceLength, depth=2, nbClasses=11, device=device)
    
    trainer = TrainerDirector.get_multiclass_trainer(model=model, traindataset=traindataset, testdataset=testdataset, 
                                                    num_classes=11, batch_size=256, device=device, nb_epochs=20,
                                                    collate_fn=collate_callable(preprocessor)
                                                    ) 
    
    trainer.summarize_model()
    
    # trainer.load_model("saves/NLP_Mixer_Binary-2022-09-12-12:44")
    
    trainer.train()
    
    trainer.save_model()
    
        
    trainer.validate()

