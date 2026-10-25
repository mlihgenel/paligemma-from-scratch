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


# GQA/MQA: K ve V'de Q'dan az head var (Gemma 2B'de 1 tane). Skoru hesaplamak için her KV head'i,
# kendisini kullanan Q head'lerinin sayısı (n_rep = n_heads / n_kv_heads) kadar tekrarlanıyor.
# Bellekte kopya tutan asıl K/V (cache) küçük kalıyor, tekrar sadece hesap anında yapılıyor
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    B, n_kv_heads, T, head_dim = x.shape
    if n_rep == 1:
        return x
    # [B, n_kv, T, hd] -> [B, n_kv, 1, T, hd] -> [B, n_kv, n_rep, T, hd]: expand veri kopyalamıyor, sadece görünüm
    x = x[:, :, None, :, :].expand(B, n_kv_heads, n_rep, T, head_dim)
    # Birleştirme: ilk n_rep head ilk KV head'in kopyası, sonraki n_rep head ikincinin... (Q head'leri de aynı sırayla gruplanıyor)
    # n_kv = 1'de (MQA) reshape de kopyalamıyor, görünüm döndürüyor. n_kv > 1'de bellek düzeni sürekli olmadığı için kopyalıyor
    return x.reshape(B, n_kv_heads * n_rep, T, head_dim)           # [B, n_heads, T, hd]


# Üretimde her adımda sadece yeni token'ın K/V'si hesaplanıyor, geçmiş token'larınki burada saklanıyor.
# Q saklanmıyor: geçmiş token'ların sorusu bir kez soruldu, cevabı zaten üretildi. Yeni token'ın Q'su
# ise bütün geçmiş K'lara bakmak zorunda. Katman başına ayrı K ve V tensörü (her katmanın K/V'si farklı)
class KVCache:
    def __init__(self):
        self.key_cache: list[torch.Tensor] = []      # katman başına [B, n_kv, T_geçmiş, hd]
        self.value_cache: list[torch.Tensor] = []

    # Kaç token görüldü. Bütün katmanlar aynı uzunlukta, 0. katmana bakmak yeterli. Boşsa 0 (prefill öncesi)
    def get_seq_length(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        return self.key_cache[0].shape[-2]

    # Yeni K/V'yi (RoPE uygulanmış) geçmişin sonuna ekle, birleşmiş halini döndür: attention bunu kullanacak.
    # Prefill'de T_yeni = prompt uzunluğu, üretimde T_yeni = 1
    def update(
        self,
        key_states: torch.Tensor,      # [B, n_kv, T_yeni, hd]
        value_states: torch.Tensor,    # [B, n_kv, T_yeni, hd]
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            # Bu katman ilk kez geliyor (prefill): saklanacak geçmiş yok, olduğu gibi sakla
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # Dizi ekseninde (-2) birleştir: [B, n_kv, T_geçmiş, hd] + [B, n_kv, T_yeni, hd] -> [B, n_kv, T_geçmiş + T_yeni, hd]
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


# Toplanan maske: izinli yerler 0, yasak yerler en küçük sayı. Token i sadece j <= i olanlara bakabilir.
# -inf yerine finfo.min: tamamen yasaklanmış bir satır -inf ile softmax'ta NaN verir, finfo.min ile eşit ağırlık verir.
# past_len: cache'te zaten duran token sayısı. Satırlar yeni T token, sütunlar geçmiş + yeni (past_len + T) token.
# Yeni token i'nin gerçek konumu past_len + i: geçmişin hepsini ve yeniler arasında kendisine kadar olanları görüyor
def make_causal_mask(T: int, dtype: torch.dtype, device: torch.device, past_len: int = 0) -> torch.Tensor:
    mask = torch.full((T, past_len + T), torch.finfo(dtype).min, dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=past_len + 1)   # j > past_len + i yasak. past_len = 0'da eski kare maske
    return mask[None, None]                           # [1, 1, T, past_len + T]: batch ve head eksenlerinde yayılıyor


# Her token, kendi Q'sunu bütün K'larla karşılaştırıp (skor) V'lerin ağırlıklı toplamını alıyor.
# Q 8 head, K/V 1 head (MQA). kv_cache verilirse geçmiş token'ların K/V'si oradan geliyor:
# Q sadece yeni token'lar için (T_yeni), K/V geçmiş + yeni bütün token'lar için (T_top = T_geçmiş + T_yeni)
class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        # head_dim config'ten okunuyor: hidden_size / n_heads değil (Gemma 2'de ikisi farklı)
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads      # n_rep: bir KV head'i kaç Q head'e hizmet ediyor
        self.scaling = self.head_dim ** -0.5

        # Q tam sayıda head üretiyor, K/V az. Burada tasarruf: k_proj ve v_proj küçük
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        # Girişi hidden_size'a eşit olmak zorunda değil: num_heads * head_dim (Gemma 2'de 8 * 256 != 2304)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,                         # [B, T, D], cache'le üretimde T = 1
        position_embeddings: tuple[torch.Tensor, torch.Tensor],   # (cos, sin), her biri [B, T, head_dim]: yeni token'ların konumları
        attention_mask: torch.Tensor | None = None,          # [B, 1, T, T_top], toplanan maske (izinli 0, yasak -inf)
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        B, T, _ = hidden_states.shape

        # Projeksiyon, sonra son boyutu head'lere böl, head eksenini öne al (head'ler birbirinden bağımsız çalışacak)
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)      # [B, n_heads, T, hd]
        k = self.k_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)   # [B, n_kv, T, hd]
        v = self.v_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)   # [B, n_kv, T, hd]

        # RoPE sadece Q ve K'ya. Konum, skorun içine girsin diye; V taşıdığı içeriğe dokunulmuyor
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Yeni K/V'yi cache'e ekle, geçmişle birleşmiş halini al. Yer önemli: RoPE'dan SONRA (her token kendi
        # konumunda bir kez döndürülüp saklanıyor), repeat_kv'den ÖNCE (cache'te n_kv head duruyor, n_heads değil)
        if kv_cache is not None:
            k, v = kv_cache.update(k, v, self.layer_idx)      # [B, n_kv, T_top, hd]

        # K/V'yi geçici olarak Q'nun head sayısına genişlet (sadece hesap anında)
        k = repeat_kv(k, self.num_kv_groups)                  # [B, n_heads, T_top, hd]
        v = repeat_kv(v, self.num_kv_groups)                  # [B, n_heads, T_top, hd]

        # Skor: [B, n_heads, T, hd] @ [B, n_heads, hd, T_top] -> [B, n_heads, T, T_top]; scores[i, j] = i. yeni token'ın j. token'a ilgisi
        # Cache yoksa T_top = T ve kare matris. Üretimde T = 1: tek satır, yeni token bütün geçmişe bakıyor
        # sqrt(head_dim)'e bölme: yoksa hd büyüdükçe skorlar büyür, softmax tek bir token'a sıkışır
        scores = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            scores = scores + attention_mask                  # yasak yerler -inf -> softmax'ta 0
        # Softmax float32'de: bf16'da toplam yuvarlama hatası birikiyor, çıkışı tekrar orijinal dtype'a çeviriyoruz
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)    # [B, n_heads, T, T_top], satırlar 1'e toplanıyor

        # Ağırlıklı V toplamı: [B, n_heads, T, T_top] @ [B, n_heads, T_top, hd] -> [B, n_heads, T, hd]
        out = torch.matmul(weights, v)
        # Head'leri tekrar yan yana birleştir: [B, n_heads, T, hd] -> [B, T, n_heads, hd] -> [B, T, n_heads * hd]
        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        return self.o_proj(out)                               # [B, T, n_heads * hd] -> [B, T, D]


# Pre-norm katman: x = x + attn(norm(x)), sonra x = x + mlp(norm(x)). Residual akışının kendisi hiç normalize edilmiyor
class GemmaDecoderLayer(nn.Module): 
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__() 
        self.config = config 
        self.hidden_size = config.hidden_size 
        
        self.self_attn = GemmaAttention(config, layer_idx)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        # Adına rağmen MLP'nin girişindeki norm (Llama'dan kalma ad). Gemma 2'de aynı ad attention çıktısına uygulanıyor
        self.post_attention_layernorm = GemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps) 
        
    def forward(
        self,
        hidden_states: torch.Tensor,                              # [B, T, D]
        position_embeddings: tuple[torch.Tensor, torch.Tensor],   # (cos, sin), her biri [B, T, head_dim]
        attention_mask: torch.Tensor | None = None,               # [B or 1, 1, T, T_top], toplanan maske
        kv_cache: KVCache | None = None,                          # sadece attention'a gidiyor, MLP token'ları karıştırmıyor
    ) -> torch.Tensor:
        # Attention bloğu. Residual norm'dan ÖNCE saklanıyor, yoksa akışın kendisi normalize edilmiş olur
        residual = hidden_states                                  # [B, T, D]
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask, kv_cache)
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
        # Düz liste parametreleri kaydetmez, Sequential ise mask/position_ids geçirmeye izin vermez
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # Residual akışı katmanlar boyunca hiç normalize edilmiyor, lm_head'den önce son bir kez
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # cos/sin her katmanda aynı (hepsinin head_dim ve base'i aynı), bu yüzden bir kez burada hesaplanıp katmanlara veriliyor
        self.rotary_emb = GemmaRotaryEmbedding(config.head_dim, base=config.rope_theta)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        inputs_embeds: torch.Tensor,                           # [B, T, D], cache'le üretimde T = 1
        attention_mask: torch.Tensor | None = None,            # [B or 1, 1, T, T_top], toplanan maske. Yoksa nedensel
        position_ids: torch.Tensor | None = None,              # [B, T]. Yoksa past_len..past_len+T-1
        scale_embeddings: bool = True,                         # False: girdi zaten otoyolun ölçeğinde (PaliGemma kendisi ölçekliyor)
        kv_cache: KVCache | None = None,                       # verilirse katmanlar geçmiş K/V'yi buradan okuyup yenisini ekliyor
    ) -> torch.Tensor:                                         # [B, T, D]
        B, T, _ = inputs_embeds.shape
        # Geçmiş uzunluğu katmanlar cache'i güncellemeden ÖNCE okunmalı: 0. katman ekledikten sonra T kadar artıyor
        past_len = kv_cache.get_seq_length() if kv_cache is not None else 0
        if position_ids is None:
            # Yeni token'lar geçmişin kaldığı yerden devam ediyor. Cache'siz 0'dan başlamak, RoPE'u yanlış konumda döndürür
            position_ids = torch.arange(past_len, past_len + T, device=inputs_embeds.device).unsqueeze(0).expand(B, -1)
        if attention_mask is None:
            attention_mask = make_causal_mask(T, inputs_embeds.dtype, inputs_embeds.device, past_len)

        hidden_states = inputs_embeds
        # Metin embedding'lerini otoyolun ölçeğine (eleman başına ~1) taşı. Ölçek hidden_states'in dtype'ında:
        # bf16'da sqrt(2048) = 45.25'e yuvarlanıyor, HF de öyle yapıyor. PaliGemma'da bunu paligemma.py yapıyor
        # (yalnızca metne: görsel özellikler projektörden zaten bu ölçekte çıkıyor), burada tekrar çarpılmamalı
        if scale_embeddings:
            normalizer = torch.tensor(self.config.hidden_size ** 0.5, dtype=hidden_states.dtype)
            hidden_states = hidden_states * normalizer

        position_embeddings = self.rotary_emb(hidden_states, position_ids)   # (cos, sin), [B, T, head_dim]
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_embeddings, attention_mask, kv_cache)

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

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        scale_embeddings: bool = True,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        hidden_states = self.model(inputs_embeds, attention_mask, position_ids, scale_embeddings, kv_cache)  # [B, T, hidden_size]
        logits = self.lm_head(hidden_states)       # [B, T, vocab_size]
        return logits.float()
    
