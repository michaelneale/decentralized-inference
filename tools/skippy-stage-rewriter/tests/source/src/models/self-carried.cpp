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

struct model_self_carried {
  struct graph {
    graph(const model_type &model);
    ggml_tensor *build_inp_embd(ggml_tensor *);
    ggml_tensor *block(ggml_tensor *, int);
    void begin_block(ggml_tensor *, int);
    void end_block(ggml_tensor *, int);
    void cb(ggml_tensor *, const char *, int);
    void ggml_build_forward_expand(void *, ggml_tensor *);
    int n_layer = 4;
    build_inputs_type build_inputs;
    graph_result *res;
    void *gf;
  };
};

model_self_carried::graph::graph(const model_type &model) {
  ggml_tensor *cur = build_inp_embd(model.tok_embd);
  for (int block_index = 0; block_index < n_layer; ++block_index) {
    cur = block(cur, block_index);
    cur = block(cur, block_index);
    cb(cur, "block_out", block_index);
  }
  ggml_build_forward_expand(gf, cur);
}
