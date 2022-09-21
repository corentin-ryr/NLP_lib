from Mixers.Datasets.DSP import IMDBSentimentAnalysis, MTOPEnglish
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import numpy as np
import math as m

def percentage_hist(l:list[int], n:int):
    
    maxVal = max(l)
    bins = np.linspace(0, maxVal, n)
    binValues = np.zeros((len(bins)))
    for value in l:
        bin = m.floor(value / maxVal * n)
        
        binValues[:bin] += 1
    
    print(bins)
    print(binValues)
    return bins, 1 - binValues / len(l)
        

dataset = IMDBSentimentAnalysis()
dataset = MTOPEnglish()

lenghts = []
nb_sample = 0
for sample, label in tqdm(dataset):
    sentence = word_tokenize(sample)
    
    lenghts.append(len(sentence))
    nb_sample += 1
    # if nb_sample > 100: break
    
print(sum(lenghts) / nb_sample)


plt.title("Lengths of the text samples")
plt.hist(lenghts, bins=50)
plt.show()

plt.title("Percentage of sample length below length.")
bins, binValues = percentage_hist(lenghts, 50)
plt.plot(bins, binValues)
plt.hlines([0.9], [0], [bins[-1]], colors=["r"])
plt.show()
