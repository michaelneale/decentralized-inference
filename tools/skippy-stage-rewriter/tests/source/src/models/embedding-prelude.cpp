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

struct model_embedding_prelude {
  struct graph {
    graph(const model_type &model);
    ggml_tensor *build_inp_embd(ggml_tensor *);
    ggml_tensor *build_inp_out_ids();
    ggml_tensor *ggml_get_rows(void *, ggml_tensor *, ggml_tensor *);
    ggml_tensor *scale(ggml_tensor *);
    ggml_tensor *block(ggml_tensor *, int);
    void begin_block(ggml_tensor *, int);
    void end_block(ggml_tensor *, int);
    void cb(ggml_tensor *, const char *, int);
    void ggml_build_forward_expand(void *, ggml_tensor *);
    int n_layer = 4;
    bool scale_embeddings = true;
    ggml_tensor *tokens;
    build_inputs_type build_inputs;
    graph_result *res;
    void *ctx0;
    void *gf;
  };
};

model_embedding_prelude::graph::graph(const model_type &model) {
  ggml_tensor *inpL = ggml_get_rows(ctx0, model.tok_embd, tokens);
  if (scale_embeddings) {
    inpL = scale(inpL);
  }
  cb(inpL, "inp_embd", -1);
  ggml_tensor *inp_out_ids = build_inp_out_ids();

  for (int il = 0; il < n_layer; ++il) {
    inpL = block(inpL, il);
    cb(inpL, "l_out", il);
  }
  ggml_build_forward_expand(gf, inpL);
}
