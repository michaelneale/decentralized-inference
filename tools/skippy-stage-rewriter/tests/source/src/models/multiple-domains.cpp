struct ggml_tensor {};
struct model_type {
  ggml_tensor *tok_embd;
};
struct domain_type {
  int n_layer;
};

struct model_multiple_domains {
  struct graph {
    graph(const model_type &model);
    ggml_tensor *build_inp_embd(ggml_tensor *);
    ggml_tensor *block(ggml_tensor *, int);
    domain_type first;
    domain_type second;
  };
};

model_multiple_domains::graph::graph(const model_type &model) {
  ggml_tensor *inpL = build_inp_embd(model.tok_embd);
  for (int il = 0; il < first.n_layer; ++il) {
    inpL = block(inpL, il);
  }
  for (int il = 0; il < second.n_layer; ++il) {
    inpL = block(inpL, il);
  }
}
