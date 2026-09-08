struct ggml_tensor {};
struct model_type {
  ggml_tensor *tok_embd;
};

struct model_two_loops {
  struct graph {
    graph(const model_type &model);
    ggml_tensor *build_inp_embd(ggml_tensor *);
    ggml_tensor *block(ggml_tensor *, int);
    int n_layer = 4;
  };
};

model_two_loops::graph::graph(const model_type &model) {
  ggml_tensor *inpL = build_inp_embd(model.tok_embd);
  for (int il = 0; il < n_layer; ++il) {
    inpL = block(inpL, il);
  }
  for (int il = 0; il < n_layer; ++il) {
    inpL = block(inpL, il);
  }
}
