struct ggml_tensor {};
struct skippy_graph_filter {
  bool enabled;
  bool include_output;
  int layer_start;
  int layer_end;
};
struct build_inputs_type { skippy_graph_filter filter; };
struct graph_result { ggml_tensor *t_embd; };
struct model_type { ggml_tensor *tok_embd; };

struct model_continue_path {
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

model_continue_path::graph::graph(const model_type &model) {
  ggml_tensor *inpL = build_inp_embd(model.tok_embd);
  for (int il = 0; il < n_layer; ++il) {
    if (il == 2) {
      continue;
    }
    inpL = block(inpL, il);
  }
}
