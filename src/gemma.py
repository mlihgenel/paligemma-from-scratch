from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

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
    
# LayerNorm'dan farkı: ortalama çıkarılmıyor ve bias yok, sadece vektörün büyüklüğü (RMS) sabitleniyor
class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Sıfırdan başlıyor: ölçek (1 + weight) olduğu için başlangıçta ölçek tam 1
        self.weight = nn.Parameter(torch.zeros(dim))

    # x / sqrt(mean(x²) + eps). Her token kendi vektörüyle normalize ediliyor: [B, T, D] -> [B, T, 1] -> broadcast
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # bf16'da kare ortalaması hassasiyet kaybeder, hesap float32'de
        output = self._norm(x.float())
        # Llama önce dtype'a döner sonra çarpar, Gemma önce float32'de çarpar sonra döner
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)
    
# GeGLU: klasik MLP'deki tek yol yerine iki paralel yol. gate "ne kadar geçsin", up "ne geçsin" kısmını taşıyor.
# Her token ayrı ayrı işleniyor, token'lar arası karışma yok (o attention'ın işi)
class GemmaMLP(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Üçü de bias'sız. Gemma 2B: 2048 -> 16384 -> 2048, katman başına ~100M parametre
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)                    # [B, T, D] -> [B, T, I]
        up = self.up_proj(x)                        # [B, T, D] -> [B, T, I]
        # Aktivasyon sadece gate'e. Gemma tanh yaklaşımıyla eğitildi, düz gelu allclose'u bozar
        h = F.gelu(gate, approximate="tanh") * up   # [B, T, I], eleman eleman çarpım
        return self.down_proj(h)                    # [B, T, I] -> [B, T, D]
        
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
    
