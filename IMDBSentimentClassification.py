from Mixers.Datasets.DSP import IMDBSentimentAnalysis
from Mixers.Models.NLPMixer.nlpmixer import NLP_Mixer
from Mixers.Trainers.classificationTrainers import ClassificationTrainer
from Mixers.Utils.helper import get_device
from Mixers.Utils.preprocessors import collate_callable, ProjectiveLayer

from torch import nn

from torchmetrics import ConfusionMatrix, Accuracy, Recall, Precision


sentenceLength = 100
textFormat = "3grammed"
useGPU = True

if __name__ == "__main__":
    device = get_device(useGPU)

    preprocessor = ProjectiveLayer(N=64, S=sentenceLength, M=1024, W=1)
    
    traindataset = IMDBSentimentAnalysis(textFormat=textFormat)
    testdataset = IMDBSentimentAnalysis(train=False, textFormat=textFormat)
    
    model = NLP_Mixer(sentenceLength=sentenceLength, depth=2, device=device)
    
    trainer = ClassificationTrainer(model, nb_epochs=40, device=device, traindataset=traindataset, testdataset=testdataset, 
                                    batch_size=256, collate_fn=collate_callable(preprocessor), num_classes=2, 
                                    metric_set={ConfusionMatrix, Accuracy, Recall, Precision}, loss=nn.BCELoss, lr=0.0001)


    trainer.summarize_model()
    
    # trainer.load_model("saves/NLP_Mixer_Binary-2022-09-12-12:44")
    
    trainer.train()
    
    trainer.save_model()
    
        
    trainer.validate()

