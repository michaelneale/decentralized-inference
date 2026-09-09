float sqrtf(float);
using int64_t = long long;
namespace std {
template <typename T> struct unique_ptr {
  T *operator->();
};
template <typename T, typename... Args> unique_ptr<T> make_unique(Args...);
template <typename T> T &&move(T &);
} // namespace std

struct ggml_tensor {
  long ne[4];
};
enum ggml_type { GGML_TYPE_F32, GGML_TYPE_I32 };

struct skippy_graph_filter {
  bool enabled;
  bool include_output;
  int layer_start;
  int layer_end;
};
struct skippy_activation_tokens {
  const int *tokens;
  long token_count;
};
struct build_inputs_type {
  skippy_graph_filter filter;
  skippy_activation_tokens activation_tokens;
};
struct ubatch_type {
  long n_tokens;
};
struct model_type {
  ggml_tensor *tok_embd;
  ggml_tensor *per_layer_tok_embd;
};
struct llm_graph_input_gemma3n_altup {
  llm_graph_input_gemma3n_altup(long, long) {}
  ggml_tensor *values;
};
struct llm_graph_input_stage_tokens {
  ggml_tensor *tokens;
};
struct graph_result {
  ggml_tensor *t_embd;
  ggml_tensor *t_skippy_gemma3n_altup;
  ggml_tensor *t_skippy_activation_input;
  ggml_tensor *t_skippy_activation_output;
  template <typename T> void add_input(std::unique_ptr<T>) {}
};

struct model_altup_sideband {
  struct graph {
    graph(const model_type &model);

    ggml_tensor *build_inp_embd(ggml_tensor *);
    ggml_tensor *build_inp_per_layer();
    ggml_tensor *build_inp_out_ids();
    ggml_tensor *project_per_layer_inputs(ggml_tensor *, ggml_tensor *);
    ggml_tensor *block(ggml_tensor *, int);
    ggml_tensor *ggml_scale(void *, ggml_tensor *, float);
    ggml_tensor *ggml_repeat_4d(void *, ggml_tensor *, long, long, long, long);
    ggml_tensor *ggml_concat(void *, ggml_tensor *, ggml_tensor *, int);
    ggml_tensor *ggml_get_rows(void *, ggml_tensor *, ggml_tensor *);
    ggml_tensor *ggml_reshape_3d(void *, ggml_tensor *, long, long, long);
    ggml_tensor *ggml_new_tensor_1d(void *, ggml_type, long);
    ggml_tensor *ggml_new_tensor_3d(void *, ggml_type, long, long, long);
    ggml_tensor *ggml_cont(void *, ggml_tensor *);
    ggml_tensor *ggml_view_2d_slice(void *, ggml_tensor *, int);
    void ggml_set_input(ggml_tensor *);
    void begin_block(ggml_tensor *, int);
    void end_block(ggml_tensor *, int);
    void cb(ggml_tensor *, const char *, int);
    void ggml_build_forward_expand(void *, ggml_tensor *);

    int n_layer = 4;
    int n_altup = 4;
    int i_altup_act = 0;
    long n_embd = 8;
    long n_tokens = 2;
    build_inputs_type build_inputs;
    ubatch_type ubatch;
    graph_result *res;
    void *ctx0;
    void *gf;
  };
};

model_altup_sideband::graph::graph(const model_type &model) {
  ggml_tensor *inpL = build_inp_embd(model.tok_embd);
  inpL = ggml_scale(ctx0, inpL, sqrtf((float)n_embd));

  ggml_tensor *inp_per_layer = build_inp_per_layer();
  inp_per_layer = project_per_layer_inputs(inpL, inp_per_layer);

  {
    ggml_tensor *repeated =
        ggml_repeat_4d(ctx0, inpL, n_embd, n_tokens, n_altup - 1, 1);
    inpL = ggml_concat(ctx0, inpL, repeated, 2);
  }

  for (int il = 0; il < n_layer; ++il) {
    ggml_tensor *cur = block(inpL, il);
    cb(cur, "l_out", il);
    inpL = cur;
  }

  inpL = ggml_view_2d_slice(ctx0, inpL, i_altup_act);
  ggml_tensor *inp_out_ids = build_inp_out_ids();
  if (inp_out_ids) {
    inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
  }
  ggml_build_forward_expand(gf, inpL);
}
