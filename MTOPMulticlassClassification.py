from mixers.datasets.DSP import MTOPEnglish
from mixers.trainers.classificationTrainers import ClassificationTrainer
from mixers.utils.preprocessors import collate_callable, ProjectiveLayer
from mixers.models.NLPMixer.nlpmixer import NLP_Mixer

import torch
from torch import nn

sentenceLength = 20
useGPU = True

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() and useGPU else "cpu")
    
    preprocessor = ProjectiveLayer(N=64, S=sentenceLength, M=1024, W=1)

    traindataset = MTOPEnglish()
    testdataset = MTOPEnglish(set="test")

    model = NLP_Mixer(sentenceLength=sentenceLength, depth=2, nbClasses=11, device=device)
  
    trainer = ClassificationTrainer(model, nb_epochs=20, device=device, traindataset=traindataset, testdataset=testdataset,
                                    batch_size=256, collate_fn=collate_callable(preprocessor), num_classes=11, loss=nn.CrossEntropyLoss)

    trainer.summarize_model()
    
    # trainer.load_model("saves/NLP_Mixer_Binary-2022-09-12-12:44")
    
    trainer.train()
    
    trainer.save_model()
    
        
    trainer.validate()

