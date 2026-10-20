import type { Arr } from "../../../lib/data";

export interface TreeNode {
  name: string; cls: string; own_params: number; params: number; children: string[];
  param_shapes: Record<string, number[]>; in?: { shape: number[]; dtype: string } | null; out?: { shape: number[]; dtype: string } | null;
}
export interface ModelTree {
  config: { vision: any; text: any; image_token_index: number; projection_dim: number };
  nodes: TreeNode[]; params_unique: number; params_state_dict: number; tied: boolean;
  keys_ours: string[]; keys_ckpt: string[]; keys_only_ours: string[]; keys_only_ckpt: string[]; seq_len: number;
}
export interface StreamRun { norms: number[]; cos_to_embed: number[]; attn_update: number[]; mlp_update: number[]; final_norm: number }
export interface GemmaToy {
  toy_config: any; seq_len: number;
  stream: { normal: StreamRun; residual_after_norm: StreamRun; post_norm: StreamRun; no_sqrt_d: StreamRun };
  stream_check_maxdiff: number;
  sqrt_d: {
    d: number; sqrt_exact: number; sqrt_bf16: number; elem_std_raw: number; elem_std_scaled: number; row_norm_raw: number[];
    named_norms: { text: string; id: number; piece: string; norm: number }[];
    roundtrip: Record<string, { maxdiff: number; rel: number }>;
    normalizer_variants: Record<string, { dtype: string; n_diff: number; n: number; maxdiff: number }>;
  };
  rms_order: { n: number; n_diff: number; maxdiff: number };
  rms_zero_init_trap: { zero_w: { ignores_weight: number }; random_w: { ignores_weight: number }; out_rms_zero_w: number };
  norm_swap: { real_w: number; zero_w: number };
  geglu: { gate: Arr; act: Arr; up: Arr; prod: Arr; frac_closed: number; frac_negative_act: number; token_independence_maxdiff: number };
  containers: Record<string, { keys: string[]; n_params: number }>;
  tying: any;
}
export interface PgSlices {
  has_lm_head_key: boolean; num_keys: number; embed_shape: number[];
  norms: { input_layernorm: Arr; post_attention_layernorm: Arr; final: Arr };
  embed_named: { text: string; id: number; piece: string }[]; embed_named_rows: Arr; embed_sample_rows: Arr;
  pos_cos: Arr; patch_filters: Arr; proj_stats: any; proj_hist: any;
}
