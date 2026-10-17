from torch import nn
from siglip import SiglipVisionConfig 

class PaliGemmaMultiModalProjector(nn.Module): 
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.linear = nn.Linear(config.hidden_size, config.projection_dim, bias=True)
        
    def forward(self, image_features): 
        hidden_states = self.linear(image_features)
        return hidden_states