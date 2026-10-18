from dataclasses import dataclass
import torch
from torch import nn

@dataclass
class GemmaConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int = 256
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_bias: bool = False
    attention_dropout: float = 0.0
    pad_token_id: int | None = None
    num_image_tokens: int = 256

class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

class GemmaForCausalLM(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    # LM head ayrı bir matris değil: embedding tablosunun kendisi (aynı Parameter nesnesi)
    def tie_weights(self) -> None:
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = self.model(inputs_embeds)  # [B, T, hidden_size]
        logits = self.lm_head(hidden_states)       # [B, T, vocab_size]
        return logits.float()
