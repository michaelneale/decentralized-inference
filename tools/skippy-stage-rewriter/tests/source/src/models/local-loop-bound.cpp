struct ggml_tensor {};
struct model_type {
  ggml_tensor *tok_embd;
};
struct hparams_type {
  int dec_n_layer;
};

struct model_local_loop_bound {
  struct graph {
    graph(const model_type &model);
    ggml_tensor *build_inp_embd(ggml_tensor *);
    ggml_tensor *block(ggml_tensor *, int);
    hparams_type hparams;
  };
};

model_local_loop_bound::graph::graph(const model_type &model) {
  ggml_tensor *inpL = build_inp_embd(model.tok_embd);
  const int dec_n_layer = hparams.dec_n_layer;
  for (int il = 0; il < dec_n_layer; ++il) {
    inpL = block(inpL, il);
  }
}
