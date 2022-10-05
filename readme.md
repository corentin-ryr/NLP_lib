 
 # NLP_Lib

 This repo contains models, datasets, trainers and various pieces of code related to my experiments in NLP.

## NLP-Mixer

The pNLP-Mixer [[1]](#1) is an architecture for language processing adapted from the MNP-Mixer [[2]](#2). It only uses linear layers and is thus very efficient. The MLP-Mixer implementation is from [Rishikesh](https://github.com/rishikksh20/MLP-Mixer-pytorch) and I implemented the projective layer used in [[1]](#1). You can run the NLP-Mixer on the MTOP and the IMDB Sentiment analysis datasets with the corresponding example scripts.

### Usage

You first need to download the [IMDB Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz) and extract it in a `data/imdb` folder.

```python
from Mixers.Datasets.DSP import IMDBSentimentAnalysis
from Mixers.NLPMixer.nlpmixer import NLP_Mixer
from Mixers.Trainers.trainerDirector import TrainerDirector
from Mixers.Utils.helper import get_device

sentenceLength = 100
textFormat = "3grammed" # You can choose between raw, tokenized and 3grammed. It is only for optimization.

device = get_device(useGPU=True)
    
traindataset = IMDBSentimentAnalysis(textFormat=textFormat, sentenceLength=sentenceLength)
testdataset = IMDBSentimentAnalysis(train=False, textFormat=textFormat, sentenceLength=sentenceLength)

model = NLP_Mixer(sentenceLength=sentenceLength, depth=2, device=device)

trainer = TrainerDirector.get_binary_trainer(model=model, traindataset=traindataset, testdataset=testdataset, batch_size=256, device=device, nb_epochs=40) 

trainer.summarize_model()

trainer.train()

trainer.validate()
```


## Installation of the package

 You can install the repo with this command:

 ```console
pip install "NLP_Lib @ git+https://github.com/corentin-ryr/NLP_lib"
```

Or you can add it to your *requirements.txt*:

```
NLP_Lib @ git+https://github.com/corentin-ryr/NLP_lib
```


## References
<a id="1">[1]</a> 
Fusco, Francesco, Damian Pascual, and Peter Staar. "pNLP-Mixer: an Efficient all-MLP Architecture for Language." arXiv preprint arXiv:2202.04350 (2022).

<a id="2">[2]</a> 
Tolstikhin, Ilya O., et al. "Mlp-mixer: An all-mlp architecture for vision." Advances in Neural Information Processing Systems 34 (2021): 24261-24272.