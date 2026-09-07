struct ggml_tensor {
  int nb[2];
};
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
  ggml_tensor *t_inp_embd;
};
struct model_type {
  ggml_tensor *tok_embd;
};

struct model_stage_zero_sideband {
  struct graph {
    graph(const model_type &model);
    ggml_tensor *build_inp_embd(ggml_tensor *);
    ggml_tensor *build_inp_out_ids();
    ggml_tensor *block(ggml_tensor *, int);
    ggml_tensor *block(ggml_tensor *, ggml_tensor *, int);
    ggml_tensor *ggml_view(void *, ggml_tensor *, int);
    void cb(ggml_tensor *, const char *, int);
    void begin_block(ggml_tensor *, int);
    void end_block(ggml_tensor *, int);
    void ggml_build_forward_expand(void *, ggml_tensor *);
    int n_layer = 4;
    int n_deepstack_layers = 2;
    build_inputs_type build_inputs;
    graph_result *res;
    void *ctx0;
    void *gf;
  };
};

model_stage_zero_sideband::graph::graph(const model_type &model) {
  ggml_tensor *inpL = build_inp_embd(model.tok_embd);
  ggml_tensor *inp_out_ids = build_inp_out_ids();

  for (int il = 0; il < n_layer; ++il) {
    inpL = block(inpL, il);
    if (il < n_deepstack_layers) {
      ggml_tensor *sideband = ggml_view(ctx0, res->t_inp_embd, il);
      inpL = block(inpL, sideband, il);
    }
    if (il == n_layer - 1 && inp_out_ids) {
      inpL = block(inpL, il);
    }
  }
  ggml_build_forward_expand(gf, inpL);
}
