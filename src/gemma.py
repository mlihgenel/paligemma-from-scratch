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


# RoPE: konum bilgisi embedding'e eklenmiyor, Q ve K vektörleri konuma göre döndürülüyor.
# head_dim boyutlu vektör head_dim/2 tane 2 boyutlu çifte ayrılıyor, her çift kendi hızında dönüyor:
# çift i, m. konumda m * theta_i açısı kadar. theta_i = base^(-2i/head_dim): ilk çiftler hızlı, son çiftler yavaş
class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        # [head_dim/2] frekans. Öğrenilen bir şey değil, sabit: persistent=False ile state_dict'e girmiyor
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    # Sadece cos/sin tablosunu üretiyor, Q/K'ye dokunmuyor. x'ten yalnızca dtype ve device alınıyor
    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Açılar float32'de: konum 8000 gibi büyük sayılara çıkınca bf16'da açı yanlış hesaplanır
        # [B, T, 1] * [head_dim/2] -> [B, T, head_dim/2], freqs[b, m, i] = m * theta_i
        freqs = position_ids[..., None].float() * self.inv_freq.to(x.device)
        # Aynı açı iki kez: rotate_half çiftleri (i, i + head_dim/2) olarak eşlediği için ikinci yarıya da lazım
        emb = torch.cat((freqs, freqs), dim=-1)                   # [B, T, head_dim]
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


# Makaledeki 2B döndürme: (a, b) -> (a cos - b sin, b cos + a sin).
# Çiftler komşu (0,1), (2,3)... değil, ilk yarı ile ikinci yarı: (x_0, x_{d/2}), (x_1, x_{d/2+1})...
# rotate_half(x) = [-x2, x1] -> x * cos + rotate_half(x) * sin bu döndürmeyi bütün çiftlere aynı anda uyguluyor
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,    # [B, n_heads, T, head_dim]
    k: torch.Tensor,    # [B, n_kv_heads, T, head_dim]
    cos: torch.Tensor,  # [B, T, head_dim]
    sin: torch.Tensor,  # [B, T, head_dim]
) -> tuple[torch.Tensor, torch.Tensor]:
    # Head boyutu ekleniyor: [B, 1, T, head_dim], her head aynı konumda aynı açıyla dönüyor
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Pre-norm katman: x = x + attn(norm(x)), sonra x = x + mlp(norm(x)). Residual akışının kendisi hiç normalize edilmiyor
class GemmaDecoderLayer(nn.Module): 
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__() 
        self.config = config 
        self.hidden_size = config.hidden_size 
        
        self.self_attn = nn.Identity()
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        # Adına rağmen MLP'nin girişindeki norm (Llama'dan kalma ad). Gemma 2'de aynı ad attention çıktısına uygulanıyor
        self.post_attention_layernorm = GemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps) 
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: 
        # Attention bloğu. Residual norm'dan ÖNCE saklanıyor, yoksa akışın kendisi normalize edilmiş olur
        residual = hidden_states                                  # [B, T, D]
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states 
        
        # MLP bloğu, aynı desen
        residual = hidden_states 
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states 
        
        return hidden_states                                      # [B, T, D]
        
class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        # Düz liste parametreleri kaydetmez, Sequential ise aşama 5'te mask/position_ids geçirmeye izin vermez
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # Residual akışı katmanlar boyunca hiç normalize edilmiyor, lm_head'den önce son bir kez
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:   # [B, T, D] -> [B, T, D]
        hidden_states = inputs_embeds
        # Embedding'leri otoyolun ölçeğine (eleman başına ~1) taşı. Görsel token'lar paligemma.py'de
        # sqrt(d)'ye bölündüğü için onlar değişmeden kalıyor. Ölçek hidden_states'in dtype'ında:
        # bf16'da sqrt(2048) = 45.25'e yuvarlanıyor, HF de öyle yapıyor
        normalizer = torch.tensor(self.config.hidden_size ** 0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states)          # aşama 5: + attention_mask, position_ids

        hidden_states = self.norm(hidden_states)
        return hidden_states


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
    
