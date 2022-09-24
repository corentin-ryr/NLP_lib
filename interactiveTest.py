from Mixers.Datasets.DSP import IMDBSentimentAnalysis
from Mixers.NLPMixer.nlpmixer import NLP_Mixer
from Mixers.Trainers.trainerDirector import TrainerDirector
from Mixers.Helper.helper import get_device
from Mixers.Inference.interactiveInference import InteractiveInferenceClassification


sentenceLength = 100
textFormat = "raw"

if __name__ == "__main__":
    useGPU = True
    device = get_device(useGPU)
    
 
    model = NLP_Mixer(sentenceLength=sentenceLength, depth=2, device=device, textFormat=textFormat)
    
    inference = InteractiveInferenceClassification(model, loadFromFile="saves/NLP_Mixer_Binary_depth2", device=device)

    inference.start()
    
