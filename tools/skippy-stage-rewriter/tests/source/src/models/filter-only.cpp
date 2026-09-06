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

struct model_filter_only {
  struct graph {
    graph(const model_type &model);

    ggml_tensor *build_inp_embd(ggml_tensor *);
    ggml_tensor *build_inp_out_ids();
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

model_filter_only::graph::graph(const model_type &model) {
  ggml_tensor *cur;
  ggml_tensor *inpL;

  const skippy_graph_filter &stage_filter = build_inputs.filter;
  const bool stage_filtered = stage_filter.enabled;
  const int il_start = stage_filtered ? stage_filter.layer_start : 0;
  const int il_end = stage_filtered ? stage_filter.layer_end : n_layer;

  inpL = build_inp_embd(stage_filtered && il_start > 0 ? nullptr : model.tok_embd);
  ggml_tensor *inp_out_ids =
      (!stage_filtered || stage_filter.include_output) ? build_inp_out_ids()
                                                       : nullptr;

  for (int il = il_start; il < il_end; ++il) {
    cur = block(inpL, il);
    if (il == il_end - 1 && inp_out_ids) {
      cur = block(cur, il);
    }

    inpL = cur;
  }

  if (stage_filtered && !stage_filter.include_output) {
    cb(inpL, "stage_boundary", il_end - 1);
    res->t_embd = inpL;
    ggml_build_forward_expand(gf, inpL);
    return;
  }

  cur = inpL;
  ggml_build_forward_expand(gf, cur);
}
