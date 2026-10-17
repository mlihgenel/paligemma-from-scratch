from dataclasses import dataclass
import torch
from torch import nn 
import torch.nn.functional as F

@dataclass
class SiglipVisionConfig: 
    num_channels: int = 3 
    hidden_size: int = 768 
    image_size: int = 224 
    patch_size: int = 16 
    intermediate_size: int = 3072 
    num_hidden_layers: int = 12 
    num_attention_heads: int = 12 
    layer_norm_eps: float = 1e-6 
    attention_dropout: float = 0.0 
    projection_dim: int = 2048
    
class SiglipVisionEmbeddings(nn.Module): 
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config 
        
        self.num_channels = config.num_channels
        self.hidden_size = config.hidden_size
        self.image_size = config.image_size 
        self.patch_size = config.patch_size
        
        self.patch_embedding = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid"
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2 
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(
            num_embeddings=self.num_positions, 
            embedding_dim=self.hidden_size
        )
        self.register_buffer(
            name="position_ids", 
            tensor=torch.arange(self.num_positions).expand(1, -1),
            persistent=False 
        )
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor: 
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(start_dim=2, end_dim=3)
        embeddings = embeddings.transpose(1, 2)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings
        
class SiglipAttention(nn.Module): 
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config 
        
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout 
        
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: 
        B, T, C = hidden_states.shape 
        k_states = self.k_proj(hidden_states) # [B, 196, 768]
        q_states = self.q_proj(hidden_states) # [B, 196, 768]
        v_states = self.v_proj(hidden_states) # [B, 196, 768] 
        
        k_states = k_states.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # [B, 12, 196, 64]
        q_states = q_states.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # [B, 12, 196, 64]
        v_states = v_states.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # [B, 12, 196, 64]
        
        wei = (q_states @ k_states.transpose(-2, -1)) * self.scale # [B, 12, 196, 196]
        wei = F.softmax(wei, dim=-1)
        wei = F.dropout(wei, p=self.dropout, training=self.training)
        
        out = wei @ v_states  # [B, 12, 196, 64]
        out = out.transpose(1, 2).contiguous()
        out = out.reshape(B, T, C)
        out = self.out_proj(out)
        return out 
        
        
class SiglipMLP(nn.Module): 
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: 
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states)
        return hidden_states
    
class SiglipEncoderLayer(nn.Module): 
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config 
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = SiglipAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: 
        residual = hidden_states 
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states 
        
        residual = hidden_states 
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states 
        
class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config 
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: 
        for layer in self.layers: 
            hidden_states = layer(hidden_states)
        return hidden_states 
    
class SiglipVisionTransformer(nn.Module): 
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config 
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor: 
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state
        