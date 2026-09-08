namespace std {
template <typename T> struct unique_ptr {
  T *operator->();
};
template <typename T, typename... Args> unique_ptr<T> make_unique(Args...) {
  return {};
}
template <typename T> T &&move(T &value) {
  return static_cast<T &&>(value);
}
} // namespace std

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
  ggml_tensor *t_skippy_activation_input;
  ggml_tensor *t_skippy_activation_output;
  template <typename T> void add_input(std::unique_ptr<T>) {}
};
struct model_type {
  ggml_tensor *tok_embd;
};
struct llm_graph_input_hyperconnection {
  llm_graph_input_hyperconnection(int, int) {}
  ggml_tensor *values;
};

enum { GGML_TYPE_F32 };

struct model_hyperconnection {
  struct graph {
    graph(const model_type &model);
    ggml_tensor *build_inp_embd(ggml_tensor *);
    ggml_tensor *build_inp_out_ids();
    ggml_tensor *ggml_reshape_3d(void *, ggml_tensor *, int, int, int);
    ggml_tensor *ggml_repeat_4d(void *, ggml_tensor *, int, int, int, int);
    ggml_tensor *ggml_new_tensor_3d(void *, int, int, int, int);
    ggml_tensor *block(ggml_tensor *, int);
    void ggml_set_input(ggml_tensor *);
    void begin_block(ggml_tensor *, int);
    void end_block(ggml_tensor *, int);
    void cb(ggml_tensor *, const char *, int);
    void ggml_build_forward_expand(void *, ggml_tensor *);
    int n_layer = 4;
    int n_embd = 8;
    int n_tokens = 2;
    int hc = 3;
    build_inputs_type build_inputs;
    graph_result *res;
    void *ctx0;
    void *gf;
  };
};

model_hyperconnection::graph::graph(const model_type &model) {
  ggml_tensor *inpL = build_inp_embd(model.tok_embd);
  cb(inpL, "model.input_embed", -1);
  ggml_build_forward_expand(gf, inpL);
  const int hc_count = hc;
  ggml_tensor *res_hc = ggml_repeat_4d(
      ctx0, ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens),
      n_embd, hc_count, n_tokens, 1);
  cb(res_hc, "hc_init", -1);

  for (int il = 0; il < n_layer; ++il) {
    res_hc = block(res_hc, il);
    cb(res_hc, "l_out", il);
  }
  ggml_build_forward_expand(gf, res_hc);
}
