# from torchtext.transforms import BERTTokenizer 
import torch

from nltk.tokenize import word_tokenize
from nltk import ngrams

import torch.nn as nn
from Mixers.NLPMixer.hashing import MultiHashing
import numpy as np


# VOCAB_FILE = "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt"

# tokenizer = BERTTokenizer(vocab_path=VOCAB_FILE, do_lower_case=True, return_tokens=True)

# tokenizer("Hello World, How are you!") # single sentence input

# tokenizer(["Hello World","How are you!"]) # batch input


def pad_list(l, length):
    l = l[:length]
    l += [""] * (length - len(l))
    return l




class collate_callable():
    def __init__(self, sentenceLength:int=200) -> None:
        self.sentenceLength = sentenceLength

    def __call__(self, data):
        data, label = list(zip(*data))
        label = torch.stack(label)

        return data, label


class ProjectiveLayer(nn.Module):
    
    def __init__(self, N:int, S:int, M:int, W:int) -> None:
        """_summary_

        Args:
            N (int): Number of hash functions
        """
        super().__init__()
        

        self.nbHashFunc = N
        self.sentenceLength = S
        self.bloomLength = M
        self.windowSize = W
        self.hashFunc = MultiHashing(self.nbHashFunc)
        
        
    def forward(self, batchSentences:list[str]) -> torch.Tensor:
                
        sentencesMinHashes = np.zeros( (len(batchSentences), self.sentenceLength, self.nbHashFunc), dtype=np.int64 )

        for idxSentence, sentence in enumerate(batchSentences):
            if type(sentence) == str:
                sentence = word_tokenize(sentence)[:self.sentenceLength]

            for idx, word in enumerate(sentence):
                if type(word) == str:
                    wordGrammed = ["".join(i) for i in ngrams(word, 3)]
                    if not wordGrammed: wordGrammed = [word]
                    word = wordGrammed

                sentencesMinHashes[idxSentence, idx] = np.min(np.array(
                    [self.hashFunc.compute_hashes(gram) for gram in word]
                    ), axis=0)
            
        floatCounter = self.counting_bloom_filter(sentencesMinHashes)
        batchmovingWindowFloatCounter = self.moving_window(floatCounter)

        return batchmovingWindowFloatCounter
    
 
    def counting_bloom_filter(self, F:np.ndarray) -> torch.Tensor:
        """_summary_
        Args:
            F (list[list[int]]): input token hashes (F). Size: nb words x nb filters
        Returns:
            torch.Tensor: Float counting tensor. Size: bloom length x max sentence length
        """
        Fp = np.remainder(F, self.bloomLength)
        CountingBloom = torch.tensor(np.apply_along_axis(lambda x: np.bincount(x, minlength=self.bloomLength), axis=2, arr=Fp))
        return torch.transpose(CountingBloom, 1, 2)
        
    
    def moving_window(self, floatCounter:torch.Tensor) -> torch.Tensor:
        l, m, s = floatCounter.shape
        movingFloatCounter = torch.zeros( (l, ( 2 * self.windowSize + 1) * m, s) )
        
        for idx, i in enumerate(range(self.windowSize, 0, -1)):
            movingFloatCounter[:, idx * m:(idx + 1) * m, i:] = floatCounter[:, :, :s - i]
                
        for i in range(self.windowSize + 1):
            movingFloatCounter[:, (i + self.windowSize) * m:(i + self.windowSize + 1) * m, :s - i] = floatCounter[:, :, i:]
        
        return movingFloatCounter
      