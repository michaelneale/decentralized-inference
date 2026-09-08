using uint32_t = unsigned;
struct ggml_tensor {};
struct skippy_graph_filter {
  bool enabled;
  bool include_output;
  int layer_start;
  int layer_end;
};
struct build_inputs_type {
  skippy_graph_filter filter;
};
struct graph_result {
  ggml_tensor *t_embd;
};
struct model_type {
  ggml_tensor *tok_embd;
};
struct hparams_type {
  bool is_recr(int) const;
  bool is_ple(uint32_t) const;
  int n_no_rope_layer_step = 2;
  int ple_n_heads = 1;
};

struct model_range_owned_inputs {
  struct graph {
    graph(const model_type &model);

    ggml_tensor *build_inp_embd(ggml_tensor *);
    ggml_tensor *build_inp_pos();
    ggml_tensor *build_inp_attn_scale();
    ggml_tensor *build_inp_ple(void *);
    ggml_tensor *build_inp_out_ids();
    ggml_tensor *block(ggml_tensor *, ggml_tensor *, int);
    void cb(ggml_tensor *, const char *, int);
    void begin_block(ggml_tensor *, int);
    void end_block(ggml_tensor *, int);
    void ggml_build_forward_expand(void *, ggml_tensor *);

    int n_layer = 4;
    hparams_type hparams;
    build_inputs_type build_inputs;
    graph_result *res;
    void *memory;
    void *gf;
  };
};

model_range_owned_inputs::graph::graph(const model_type &model) {
  ggml_tensor *inpL = build_inp_embd(model.tok_embd);
  ggml_tensor *inp_pos = build_inp_pos();
  ggml_tensor *inp_attn_scale = build_inp_attn_scale();
  ggml_tensor *inp_out_ids = build_inp_out_ids();

  ggml_tensor *ple_emb = nullptr;
  if (hparams.ple_n_heads > 0) {
    ple_emb = build_inp_ple(memory);
    ggml_build_forward_expand(gf, ple_emb);
  }

  for (int il = 0; il < n_layer; ++il) {
    if (hparams.is_ple(il)) {
      inpL = block(inpL, ple_emb, il);
    }
    if (hparams.is_recr(il)) {
      inpL = block(inpL, nullptr, il);
    } else {
      inpL = block(inpL, inp_pos, il);
    }
    if (hparams.n_no_rope_layer_step == 0 ||
        (il + 1) % hparams.n_no_rope_layer_step == 0) {
      inpL = block(inpL, inp_attn_scale, il);
    }
    if (il == n_layer - 1 && inp_out_ids) {
      inpL = block(inpL, inp_out_ids, il);
    }
  }

  ggml_build_forward_expand(gf, inpL);
}
