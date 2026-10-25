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
    
# Unlike LayerNorm: no mean subtraction and no bias, only the vector's magnitude (RMS) is fixed
class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Starts at zero: the scale is (1 + weight), so it is exactly 1 at init
        self.weight = nn.Parameter(torch.zeros(dim))

    # x / sqrt(mean(x²) + eps). Each token is normalized by its own vector: [B, T, D] -> [B, T, 1] -> broadcast
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The mean of squares loses precision in bf16, so compute in float32
        output = self._norm(x.float())
        # Llama casts back to dtype and then multiplies, Gemma multiplies in float32 and then casts back
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)
    
# GeGLU: two parallel paths instead of the single path of a classic MLP. gate carries "how much passes", up carries "what passes".
# Each token is processed on its own, nothing mixes across tokens (that is attention's job)
class GemmaMLP(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # All three without bias. Gemma 2B: 2048 -> 16384 -> 2048, ~100M parameters per layer
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)                    # [B, T, D] -> [B, T, I]
        up = self.up_proj(x)                        # [B, T, D] -> [B, T, I]
        # Activation only on gate. Gemma was trained with the tanh approximation, exact gelu breaks allclose
        h = F.gelu(gate, approximate="tanh") * up   # [B, T, I], elementwise product
        return self.down_proj(h)                    # [B, T, I] -> [B, T, D]


# RoPE: position is not added to the embedding, Q and K vectors are rotated by position instead.
# A head_dim vector is split into head_dim/2 two-dimensional pairs, each pair rotates at its own speed:
# pair i at position m rotates by m * theta_i. theta_i = base^(-2i/head_dim): first pairs fast, last pairs slow
class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        # [head_dim/2] frequencies. Not learned, fixed: persistent=False keeps them out of the state_dict
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    # Only builds the cos/sin table, does not touch Q/K. Only dtype and device are taken from x
    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Angles in float32: for large positions like 8000 the angle comes out wrong in bf16
        # [B, T, 1] * [head_dim/2] -> [B, T, head_dim/2], freqs[b, m, i] = m * theta_i
        freqs = position_ids[..., None].float() * self.inv_freq.to(x.device)
        # Same angle twice: rotate_half pairs (i, i + head_dim/2), so the second half needs it too
        emb = torch.cat((freqs, freqs), dim=-1)                   # [B, T, head_dim]
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


# The 2D rotation from the paper: (a, b) -> (a cos - b sin, b cos + a sin).
# Pairs are not neighbors (0,1), (2,3)... but first half with second half: (x_0, x_{d/2}), (x_1, x_{d/2+1})...
# rotate_half(x) = [-x2, x1] -> x * cos + rotate_half(x) * sin applies this rotation to all pairs at once
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
    # Add a head axis: [B, 1, T, head_dim], every head rotates by the same angle at the same position
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# GQA/MQA: K and V have fewer heads than Q (1 in Gemma 2B). To compute scores, each KV head is repeated
# once per Q head that uses it (n_rep = n_heads / n_kv_heads).
# The stored K/V (the cache) stays small, the repeat only happens at compute time
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    B, n_kv_heads, T, head_dim = x.shape
    if n_rep == 1:
        return x
    # [B, n_kv, T, hd] -> [B, n_kv, 1, T, hd] -> [B, n_kv, n_rep, T, hd]: expand copies no data, it is just a view
    x = x[:, :, None, :, :].expand(B, n_kv_heads, n_rep, T, head_dim)
    # Merge: the first n_rep heads copy the first KV head, the next n_rep the second... (Q heads are grouped in the same order)
    # With n_kv = 1 (MQA) reshape returns a view too. With n_kv > 1 the memory is not contiguous, so it copies
    return x.reshape(B, n_kv_heads * n_rep, T, head_dim)           # [B, n_heads, T, hd]


# During generation each step computes K/V only for the new token, the past tokens' K/V are kept here.
# Q is not kept: past tokens already asked their question and got their answer. The new token's Q,
# however, has to look at all past K's. One K and one V tensor per layer (every layer has its own K/V)
class KVCache:
    def __init__(self):
        self.key_cache: list[torch.Tensor] = []      # per layer [B, n_kv, T_past, hd]
        self.value_cache: list[torch.Tensor] = []

    # How many tokens have been seen. All layers have the same length, so layer 0 is enough. 0 if empty (before prefill)
    def get_seq_length(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        return self.key_cache[0].shape[-2]

    # Append the new K/V (with RoPE applied) to the past and return the concatenation: attention uses it.
    # At prefill T_new = prompt length, during generation T_new = 1
    def update(
        self,
        key_states: torch.Tensor,      # [B, n_kv, T_new, hd]
        value_states: torch.Tensor,    # [B, n_kv, T_new, hd]
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            # First call for this layer (prefill): there is no past, store as is
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # Concatenate on the sequence axis (-2): [B, n_kv, T_past, hd] + [B, n_kv, T_new, hd] -> [B, n_kv, T_past + T_new, hd]
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


# Additive mask: allowed positions 0, blocked positions the smallest number. Token i may only look at j <= i.
# finfo.min instead of -inf: a fully blocked row gives NaN in softmax with -inf, equal weights with finfo.min.
# past_len: number of tokens already in the cache. Rows are the T new tokens, columns past + new (past_len + T).
# New token i sits at position past_len + i: it sees all of the past and the new tokens up to itself
def make_causal_mask(T: int, dtype: torch.dtype, device: torch.device, past_len: int = 0) -> torch.Tensor:
    mask = torch.full((T, past_len + T), torch.finfo(dtype).min, dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=past_len + 1)   # j > past_len + i blocked. past_len = 0 gives the square mask
    return mask[None, None]                           # [1, 1, T, past_len + T]: broadcast over batch and heads


# Each token compares its Q with all K's (scores) and takes a weighted sum of the V's.
# Q has 8 heads, K/V 1 head (MQA). With a kv_cache, the past tokens' K/V come from it:
# Q only for the new tokens (T_new), K/V for all past + new tokens (T_all = T_past + T_new)
class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        # head_dim is read from the config, not hidden_size / n_heads (they differ in Gemma 2)
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads      # n_rep: how many Q heads share one KV head
        self.scaling = self.head_dim ** -0.5

        # Q gets the full number of heads, K/V fewer. The saving: k_proj and v_proj are small
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        # Its input need not equal hidden_size: num_heads * head_dim (8 * 256 != 2304 in Gemma 2)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,                         # [B, T, D], T = 1 when generating with a cache
        position_embeddings: tuple[torch.Tensor, torch.Tensor],   # (cos, sin), each [B, T, head_dim]: positions of the new tokens
        attention_mask: torch.Tensor | None = None,          # [B, 1, T, T_all], additive mask (allowed 0, blocked -inf)
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        B, T, _ = hidden_states.shape

        # Project, split the last dim into heads, move the head axis forward (heads work independently)
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)      # [B, n_heads, T, hd]
        k = self.k_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)   # [B, n_kv, T, hd]
        v = self.v_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)   # [B, n_kv, T, hd]

        # RoPE only on Q and K, so position enters the scores; the content carried by V is left alone
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Add the new K/V to the cache and get it back joined with the past. Placement matters: AFTER RoPE (each token is
        # rotated once at its own position and stored), BEFORE repeat_kv (the cache holds n_kv heads, not n_heads)
        if kv_cache is not None:
            k, v = kv_cache.update(k, v, self.layer_idx)      # [B, n_kv, T_all, hd]

        # Temporarily expand K/V to Q's head count (only at compute time)
        k = repeat_kv(k, self.num_kv_groups)                  # [B, n_heads, T_all, hd]
        v = repeat_kv(v, self.num_kv_groups)                  # [B, n_heads, T_all, hd]

        # Scores: [B, n_heads, T, hd] @ [B, n_heads, hd, T_all] -> [B, n_heads, T, T_all]; scores[i, j] = how much new token i attends to token j
        # Without a cache T_all = T and the matrix is square. When generating T = 1: one row, the new token looks at the whole past
        # Divide by sqrt(head_dim): otherwise scores grow with hd and softmax collapses onto a single token
        scores = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            scores = scores + attention_mask                  # blocked positions -inf -> 0 after softmax
        # Softmax in float32: rounding error accumulates in bf16; the output is cast back to the original dtype
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)    # [B, n_heads, T, T_all], rows sum to 1

        # Weighted sum of V: [B, n_heads, T, T_all] @ [B, n_heads, T_all, hd] -> [B, n_heads, T, hd]
        out = torch.matmul(weights, v)
        # Put the heads back side by side: [B, n_heads, T, hd] -> [B, T, n_heads, hd] -> [B, T, n_heads * hd]
        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        return self.o_proj(out)                               # [B, T, n_heads * hd] -> [B, T, D]


# Pre-norm layer: x = x + attn(norm(x)), then x = x + mlp(norm(x)). The residual stream itself is never normalized
class GemmaDecoderLayer(nn.Module): 
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__() 
        self.config = config 
        self.hidden_size = config.hidden_size 
        
        self.self_attn = GemmaAttention(config, layer_idx)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        # Despite the name, this is the norm before the MLP (a name inherited from Llama). In Gemma 2 the same name is applied to the attention output
        self.post_attention_layernorm = GemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps) 
        
    def forward(
        self,
        hidden_states: torch.Tensor,                              # [B, T, D]
        position_embeddings: tuple[torch.Tensor, torch.Tensor],   # (cos, sin), each [B, T, head_dim]
        attention_mask: torch.Tensor | None = None,               # [B or 1, 1, T, T_all], additive mask
        kv_cache: KVCache | None = None,                          # only attention uses it, the MLP does not mix tokens
    ) -> torch.Tensor:
        # Attention block. The residual is saved BEFORE the norm, otherwise the stream itself would get normalized
        residual = hidden_states                                  # [B, T, D]
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask, kv_cache)
        hidden_states = residual + hidden_states 
        
        # MLP block, same pattern
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
        # A plain list does not register parameters, and Sequential cannot pass mask/position_ids
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # The residual stream is never normalized across the layers, only once at the end before lm_head
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # cos/sin are the same in every layer (same head_dim and base), so they are computed once here and passed down
        self.rotary_emb = GemmaRotaryEmbedding(config.head_dim, base=config.rope_theta)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        inputs_embeds: torch.Tensor,                           # [B, T, D], T = 1 when generating with a cache
        attention_mask: torch.Tensor | None = None,            # [B or 1, 1, T, T_all], additive mask. Causal if None
        position_ids: torch.Tensor | None = None,              # [B, T]. past_len..past_len+T-1 if None
        scale_embeddings: bool = True,                         # False: input is already at the residual stream's scale (PaliGemma scales it itself)
        kv_cache: KVCache | None = None,                       # if given, layers read the past K/V from it and append the new ones
    ) -> torch.Tensor:                                         # [B, T, D]
        B, T, _ = inputs_embeds.shape
        # Read the past length BEFORE the layers update the cache: it grows by T once layer 0 appends
        past_len = kv_cache.get_seq_length() if kv_cache is not None else 0
        if position_ids is None:
            # New tokens continue where the past left off. Starting from 0 would rotate RoPE at the wrong position
            position_ids = torch.arange(past_len, past_len + T, device=inputs_embeds.device).unsqueeze(0).expand(B, -1)
        if attention_mask is None:
            attention_mask = make_causal_mask(T, inputs_embeds.dtype, inputs_embeds.device, past_len)

        hidden_states = inputs_embeds
        # Bring text embeddings to the residual stream's scale (~1 per element). The scale is in hidden_states' dtype:
        # in bf16 sqrt(2048) rounds to 45.25, and HF does the same. In PaliGemma this is done by paligemma.py
        # (text only: image features already leave the projector at this scale), so it must not be applied again here
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

    # The LM head is not a separate matrix: it is the embedding table itself (the same Parameter object)
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
    
