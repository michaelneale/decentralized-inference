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

struct model_combined_context {
  struct graph {
    graph(const model_type &model);

    ggml_tensor *build_inp_embd(ggml_tensor *);
    ggml_tensor *build_inp_out_ids();
    ggml_tensor *block(ggml_tensor *, int);
    void use_context(void *);
    void begin_block(ggml_tensor *, int);
    void end_block(ggml_tensor *, int);
    void cb(ggml_tensor *, const char *, int);
    void ggml_build_forward_expand(void *, ggml_tensor *);

    bool is_mtp = false;
    int n_layer = 4;
    void *ctx_other = nullptr;
    build_inputs_type build_inputs;
    graph_result *res;
    void *gf;
  };
};

model_combined_context::graph::graph(const model_type &model) {
  if (is_mtp) {
    use_context(ctx_other);
    ggml_tensor *sidecar_out_ids = build_inp_out_ids();
    use_context(sidecar_out_ids);
    return;
  }

  ggml_tensor *inpL = build_inp_embd(model.tok_embd);

  for (int il = 0; il < n_layer; ++il) {
    ggml_tensor *cur = block(inpL, il);
    inpL = cur;
  }

  ggml_tensor *inp_out_ids = build_inp_out_ids();
  use_context(inp_out_ids);
  ggml_build_forward_expand(gf, inpL);
}
