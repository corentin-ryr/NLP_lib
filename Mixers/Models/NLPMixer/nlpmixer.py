import torch
import torch.nn as nn

from Mixers.Models.MLPMixer.mlpmixer import MixerBlock

        
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