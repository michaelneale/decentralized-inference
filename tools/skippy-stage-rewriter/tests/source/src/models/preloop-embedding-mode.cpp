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
struct ubatch_type {
  ggml_tensor *embd;
};

struct model_preloop_embedding_mode {
  struct graph {
    graph(const model_type &model);
    ggml_tensor *build_inp_embd(ggml_tensor *);
    ggml_tensor *build_inp_out_ids();
    ggml_tensor *block(ggml_tensor *, int);
    void fail_embedding_mode();
    void begin_block(ggml_tensor *, int);
    void end_block(ggml_tensor *, int);
    void cb(ggml_tensor *, const char *, int);
    void ggml_build_forward_expand(void *, ggml_tensor *);
    int n_layer = 4;
    bool use_mrope = false;
    ubatch_type ubatch;
    build_inputs_type build_inputs;
    graph_result *res;
    void *gf;
  };
};

model_preloop_embedding_mode::graph::graph(const model_type &model) {
  ggml_tensor *inpL = build_inp_embd(model.tok_embd);
  if (ubatch.embd && !use_mrope) {
    fail_embedding_mode();
  }
  ggml_tensor *inp_out_ids = build_inp_out_ids();

  for (int il = 0; il < n_layer; ++il) {
    inpL = block(inpL, il);
    if (il == n_layer - 1 && inp_out_ids) {
      inpL = block(inpL, il);
    }
  }
  ggml_build_forward_expand(gf, inpL);
}
