from typing import Union

import torch
import torch.nn as nn

from nltk.tokenize import word_tokenize
from nltk import ngrams

from Mixers.Models.NLPMixer.hashing import MultiHashing
from Mixers.Models.MLPMixer.mlpmixer import MixerBlock
from Mixers.Helper.helper import pad_list

from einops.layers.torch import Rearrange
import numpy as np


class ProjectiveLayer():
    
    def __init__(self, N:int, S:int, M:int, W:int) -> None:
        """_summary_

        Args:
            N (int): Number of hash functions
        """
        self.nbHashFunc = N
        self.sentenceLength = S
        self.bloomLength = M
        self.windowSize = W
        self.hashFunc = MultiHashing(self.nbHashFunc)

        self.rearrange = Rearrange('b n d -> b d n')
        
        
    def __call__(self, batchSentences:list[str]) -> torch.Tensor:
                
        sentencesMinHashes = np.zeros( (len(batchSentences), self.sentenceLength, self.nbHashFunc), dtype=np.int64 )

        for idxSentence, sentence in enumerate(batchSentences):
            if type(sentence) == str:
                sentence = word_tokenize(sentence)

            for idx, word in enumerate(pad_list(sentence, self.sentenceLength)):
                if type(word) == str:
                    wordGrammed = ["".join(i) for i in ngrams(word, 3)]
                    if not wordGrammed: wordGrammed = [word]
                    word = wordGrammed

                sentencesMinHashes[idxSentence, idx] = np.min(np.array(
                    [self.hashFunc.compute_hashes(gram) for gram in word]
                    ), axis=0)
            
        floatCounter = self.counting_bloom_filter(sentencesMinHashes)
        batchmovingWindowFloatCounter = self.moving_window(floatCounter)

        return self.rearrange(batchmovingWindowFloatCounter)
    
 
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
        
        
class NLP_Mixer(nn.Module):
    
    def __init__(self, sentenceLength:int=100, bloomLength:int=1024, windowSize:int=1, bottleneckParam:int=256, 
                mixerTokenDim:int=256, mixerChannelDim:int=256, depth:int=2, nbClasses:int=None, device='cpu') -> None:
        super().__init__()

        self.sentenceLength = sentenceLength
        self.bloomLength = bloomLength
        self.windowSize = windowSize
        self.bottleneckParam = bottleneckParam
        self.mixerTokenDim = mixerTokenDim
        self.mixerChannelDim = mixerChannelDim
        self.depth = depth
        self.nbClasses = nbClasses if nbClasses else 1
        self.device = device
        
        h_proj, w_proj = (2 * self.windowSize + 1) * self.bloomLength, self.bottleneckParam
        
        self.bottleneck = nn.Linear( h_proj, w_proj )

        self.mixer_blocks = []
        for _ in range(self.depth):
            self.mixer_blocks.append(MixerBlock(self.bottleneckParam, self.sentenceLength, self.mixerTokenDim, self.mixerChannelDim))
        self.mixer_blocks = nn.Sequential(*self.mixer_blocks)

        self.layer_norm = nn.LayerNorm(self.bottleneckParam)

        self.mlp_head = nn.Linear(self.bottleneckParam, self.nbClasses)

            
    def forward(self, x:torch.Tensor):        
        
        x = self.bottleneck(x)
        x = self.mixer_blocks(x)
        x = self.layer_norm(x)
        x = torch.mean(x, dim=1)
        x = self.mlp_head(x)

        final = torch.softmax(x, 1) if self.nbClasses > 2 else torch.sigmoid(torch.squeeze(x))
        return final