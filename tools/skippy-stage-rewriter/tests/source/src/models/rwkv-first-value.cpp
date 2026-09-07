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
enum { GGML_TYPE_F32 };
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
  ggml_tensor *t_skippy_rwkv7_v_first;
  template <typename T> void add_input(std::unique_ptr<T>) {}
};
struct model_type {
  ggml_tensor *tok_embd;
};
struct llm_graph_input_rwkv7_v_first {
  explicit llm_graph_input_rwkv7_v_first(int) {}
  ggml_tensor *values;
};

struct model_rwkv_first_value {
  struct graph {
    graph(const model_type &model);
    ggml_tensor *build_inp_embd(ggml_tensor *);
    ggml_tensor *build_inp_out_ids();
    ggml_tensor *build_rwkv7_time_mix(ggml_tensor *, ggml_tensor *,
                                      ggml_tensor *, ggml_tensor *&, int);
    ggml_tensor *block(ggml_tensor *, int);
    ggml_tensor *ggml_new_tensor_2d(void *, int, int, int);
    void ggml_set_input(ggml_tensor *);
    void begin_block(ggml_tensor *, int);
    void end_block(ggml_tensor *, int);
    void cb(ggml_tensor *, const char *, int);
    void ggml_build_forward_expand(void *, ggml_tensor *);
    int n_layer = 4;
    int n_embd = 8;
    int n_tokens = 2;
    build_inputs_type build_inputs;
    graph_result *res;
    void *ctx0;
    void *gf;
  };
};

model_rwkv_first_value::graph::graph(const model_type &model) {
  ggml_tensor *inpL = build_inp_embd(model.tok_embd);
  ggml_tensor *v_first = nullptr;
  ggml_tensor *inp_out_ids = build_inp_out_ids();

  for (int il = 0; il < n_layer; ++il) {
    inpL = build_rwkv7_time_mix(nullptr, inpL, nullptr, v_first, il);
    inpL = block(inpL, il);
    if (il == n_layer - 1 && inp_out_ids) {
      inpL = block(inpL, il);
    }
  }
  ggml_build_forward_expand(gf, inpL);
}
