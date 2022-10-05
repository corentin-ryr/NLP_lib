from typing import Union

import torch
import torch.nn as nn

from Mixers.MLPMixer.mlpmixer import MixerBlock
from Mixers.Utils.preprocessors import ProjectiveLayer
from einops.layers.torch import Rearrange
  
        
class NLP_Mixer(nn.Module):
    
    def __init__(self, nbHashFunc:int=64, sentenceLength:int=100, bloomLength:int=1024, windowSize:int=1, 
                 bottleneckParam:int=256, mixerTokenDim:int=256, mixerChannelDim:int=256, depth:int=2, nbClasses:int=None, applyPreprocessing:bool=True, device='cpu') -> None:
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

        self.projectiveLayer = torch.nn.Sequential(
            ProjectiveLayer(self.nbHashFunc, self.sentenceLength, self.bloomLength, self.windowSize),
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
        
        if self.applyPreprocessing:
            x = self.projectiveLayer(x).to(self.device)

        x = self.bottleneck(x)
        
        x = self.mixer_blocks(x)
        x = self.layer_norm(x)
        x = torch.mean(x, dim=1)
        x = self.mlp_head(x)

        final = torch.softmax(x, 1) if self.nbClasses > 2 else torch.sigmoid(torch.squeeze(x))
        return final