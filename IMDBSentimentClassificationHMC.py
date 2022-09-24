from Mixers.Datasets.DSP import IMDBSentimentAnalysis
from Mixers.Models.NLPMixer.nlpmixer import NLP_Mixer, ProjectiveLayer
from Mixers.Trainers.trainerDirector import TrainerDirector
from Mixers.Helper.helper import collate_callable, get_device



sentenceLength = 100
textFormat = "raw"

if __name__ == "__main__":
    useGPU = False
    device = get_device(useGPU)

    preprocessor = ProjectiveLayer(N=64, S=sentenceLength, M=1024, W=1)

    traindataset = IMDBSentimentAnalysis(textFormat=textFormat, sentenceLength=sentenceLength, limit=10)
    testdataset = IMDBSentimentAnalysis(train=False, textFormat=textFormat, sentenceLength=sentenceLength)

    model = NLP_Mixer(sentenceLength=200, depth=2)
    
    trainer = TrainerDirector.get_hmc_trainer(model=model, traindataset=traindataset, testdataset=testdataset, 
                                            device=device, batch_size=256, 
                                            collate_fn=collate_callable(traindataset.sentenceLength, preprocessor)
                                            ) 
    
    trainer.summarize_model()
    
    # trainer.load_model("saves/NLP_Mixer_Binary-2022-09-12-12:44")
    
    trainer.train()
    
    # trainer.save_model()
    
    trainer.validate(light=True)

