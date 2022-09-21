from typing import Union
import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize
from nltk import ngrams

from Mixers.NLPMixer.hashing import MultiHashing
from Mixers.MLPMixer.mlpmixer import MixerBlock

from einops.layers.torch import Rearrange
import time
import numpy as np


class ProjectiveLayer(nn.Module):
    
    def __init__(self, N:int, S:int, M:int, W:int, textFormat="raw") -> None:
        """_summary_

        Args:
            N (int): Number of hash functions
        """
        super().__init__()
        
        self.textFormat = textFormat

        self.nbHashFunc = N
        self.sentenceLength = S
        self.bloomLength = M
        self.windowSize = W
        self.hashFunc = MultiHashing(self.nbHashFunc)
        
        
    def forward(self, batchSentences:list[str]) -> torch.Tensor:
                
        sentencesMinHashes = np.zeros( (len(batchSentences), self.sentenceLength, self.nbHashFunc), dtype=np.int64 )

        if self.textFormat == "raw":
            for idxSentence, sentence in enumerate(batchSentences):            
                for idx, word in enumerate(word_tokenize(sentence)[:self.sentenceLength]):
                    sentencesMinHashes[idxSentence, idx] = np.min(np.array(
                        [self.hashFunc.compute_hashes("".join(i)) for i in ngrams(word, 3, pad_right=True, right_pad_symbol="")]
                        ), axis=0)

        elif self.textFormat == "tokenized":
            for idxSentence, sentence in enumerate(batchSentences):            
                for idx, word in enumerate(sentence[:self.sentenceLength]):
                    sentencesMinHashes[idxSentence, idx] = np.min(np.array(
                        [self.hashFunc.compute_hashes("".join(i)) for i in ngrams(word, 3, pad_right=True, right_pad_symbol="")]
                        ), axis=0)

        elif self.textFormat == "3grammed":
            for idxSentence, sentence in enumerate(batchSentences):
                for idx, word in enumerate(sentence):
                    sentencesMinHashes[idxSentence, idx] = np.min(np.array(
                        [self.hashFunc.compute_hashes(gram) for gram in word]
                        ), axis=0)
        else: raise NotImplementedError("Use of of the supported kind of text.")
        
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
        
        
class NLP_Mixer(nn.Module):
    
    def __init__(self, nbHashFunc:int=64, sentenceLength:int=100, bloomLength:int=1024, windowSize:int=1, 
                 bottleneckParam:int=256, mixerTokenDim:int=256, mixerChannelDim:int=256, depth:int=2, nbClasses:int=None, applyPreprocessing:bool=True, device='cpu', textFormat:str="raw") -> None:
        super().__init__()

        self.nbHashFunc = nbHashFunc
        self.sentenceLength = sentenceLength
        self.bloomLength = bloomLength
        self.windowSize = windowSize
        self.bottleneckParam = bottleneckParam
        self.mixerTokenDim = mixerTokenDim
        self.mixerChannelDim = mixerChannelDim
        self.depth = depth
        self.nbClasses = nbClasses if nbClasses else 1
        self.applyPreprocessing = applyPreprocessing
        self.device = device
        self.textFormat = textFormat

        self.projectiveLayer = torch.nn.Sequential(
            ProjectiveLayer(self.nbHashFunc, self.sentenceLength, self.bloomLength, self.windowSize, textFormat),
            Rearrange('b n d -> b d n')
        )
        
        h_proj, w_proj = (2 * self.windowSize + 1) * self.bloomLength, self.bottleneckParam
        
        self.bottleneck = nn.Linear( h_proj, w_proj )

        self.mixer_blocks = []
        for _ in range(self.depth):
            self.mixer_blocks.append(MixerBlock(self.bottleneckParam, self.sentenceLength, self.mixerTokenDim, self.mixerChannelDim))
        self.mixer_blocks = nn.Sequential(*self.mixer_blocks)

        self.layer_norm = nn.LayerNorm(self.bottleneckParam)

        self.mlp_head = nn.Linear(self.bottleneckParam, self.nbClasses)
            
    def forward(self, x:Union[list[str], torch.Tensor]):
        startime = time.time()
        
        if self.applyPreprocessing:
            x = self.projectiveLayer(x).to(self.device)

        print(f"Projection time: {time.time() - startime}")
        startime = time.time()

        x = self.bottleneck(x)        
        
        x = self.mixer_blocks(x)
        x = self.layer_norm(x)
        x = torch.mean(x, dim=1)
        
        final = torch.softmax(self.mlp_head(x), 1) if self.nbClasses >= 2 else torch.squeeze(self.mlp_head(x))
        
        print(f"Network time: {time.time() - startime}")
        return final