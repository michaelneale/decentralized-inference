struct ggml_tensor {};
struct model_type {};

struct model_auxiliary {
  struct graph_mtp {
    graph_mtp(const model_type &model);
    void consume(ggml_tensor *);
    ggml_tensor *t_h_nextn;
  };
};

model_auxiliary::graph_mtp::graph_mtp(const model_type &) {
  ggml_tensor *cur = t_h_nextn;
  consume(cur);
}
