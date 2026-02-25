//! Raw FFI to our llama.cpp fork.
//!
//! Thin unsafe layer — the public API is in lib.rs.
//! Covers only what we need: model load, chat template, tokenize, decode, sample.

#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_float};

// Opaque types
pub enum llama_model {}
pub enum llama_context {}
pub enum llama_sampler {}
pub enum llama_vocab {}
pub enum llama_memory_i {}
pub type llama_memory_t = *mut llama_memory_i;

pub type llama_token = i32;
pub type llama_pos = i32;
pub type llama_seq_id = i32;

#[repr(C)]
pub struct llama_batch {
    pub n_tokens: i32,
    pub token: *mut llama_token,
    pub embd: *mut c_float,
    pub pos: *mut llama_pos,
    pub n_seq_id: *mut i32,
    pub seq_id: *mut *mut llama_seq_id,
    pub logits: *mut i8,
}

// Params structs — opaque padding, we only set fields via defaults + overrides
#[repr(C)]
pub struct llama_model_params {
    _data: [u8; 256],
}

#[repr(C)]
pub struct llama_context_params {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    _rest: [u8; 200],
}

#[repr(C)]
pub struct llama_sampler_chain_params {
    pub no_perf: bool,
}

#[repr(C)]
pub struct llama_chat_message {
    pub role: *const c_char,
    pub content: *const c_char,
}

#[link(name = "llama", kind = "static")]
#[link(name = "ggml", kind = "static")]
#[link(name = "ggml-base", kind = "static")]
#[link(name = "ggml-cpu", kind = "static")]
#[link(name = "ggml-metal", kind = "static")]
#[link(name = "ggml-blas", kind = "static")]
#[link(name = "ggml-rpc", kind = "static")]
#[link(name = "c++")]
extern "C" {
    // Backend
    pub fn llama_backend_init();

    // Model
    pub fn llama_model_default_params() -> llama_model_params;
    pub fn llama_model_load_from_file(
        path: *const c_char,
        params: llama_model_params,
    ) -> *mut llama_model;
    pub fn llama_model_free(model: *mut llama_model);
    pub fn llama_model_chat_template(
        model: *const llama_model,
        name: *const c_char,
    ) -> *const c_char;

    // Vocab
    pub fn llama_model_get_vocab(model: *const llama_model) -> *const llama_vocab;
    pub fn llama_vocab_is_eog(vocab: *const llama_vocab, token: llama_token) -> bool;

    // Tokenize / detokenize
    pub fn llama_tokenize(
        vocab: *const llama_vocab,
        text: *const c_char,
        text_len: i32,
        tokens: *mut llama_token,
        n_tokens_max: i32,
        add_special: bool,
        parse_special: bool,
    ) -> i32;
    pub fn llama_token_to_piece(
        vocab: *const llama_vocab,
        token: llama_token,
        buf: *mut c_char,
        length: i32,
        lstrip: i32,
        special: bool,
    ) -> i32;

    // Context
    pub fn llama_context_default_params() -> llama_context_params;
    pub fn llama_init_from_model(
        model: *mut llama_model,
        params: llama_context_params,
    ) -> *mut llama_context;
    pub fn llama_free(ctx: *mut llama_context);
    pub fn llama_n_ctx(ctx: *const llama_context) -> u32;

    // KV cache
    pub fn llama_get_memory(ctx: *const llama_context) -> llama_memory_t;
    pub fn llama_memory_clear(mem: llama_memory_t, data: bool);

    // Decode
    pub fn llama_decode(ctx: *mut llama_context, batch: llama_batch) -> i32;
    pub fn llama_batch_get_one(tokens: *const llama_token, n_tokens: i32) -> llama_batch;

    // Sampling
    pub fn llama_sampler_chain_default_params() -> llama_sampler_chain_params;
    pub fn llama_sampler_chain_init(params: llama_sampler_chain_params) -> *mut llama_sampler;
    pub fn llama_sampler_chain_add(chain: *mut llama_sampler, smpl: *mut llama_sampler);
    pub fn llama_sampler_free(smpl: *mut llama_sampler);
    pub fn llama_sampler_init_temp(temp: c_float) -> *mut llama_sampler;
    pub fn llama_sampler_init_top_p(p: c_float, min_keep: usize) -> *mut llama_sampler;
    pub fn llama_sampler_init_min_p(p: c_float, min_keep: usize) -> *mut llama_sampler;
    pub fn llama_sampler_init_dist(seed: u32) -> *mut llama_sampler;
    pub fn llama_sampler_sample(
        smpl: *mut llama_sampler,
        ctx: *mut llama_context,
        idx: i32,
    ) -> llama_token;

    // Chat template
    pub fn llama_chat_apply_template(
        tmpl: *const c_char,
        msgs: *const llama_chat_message,
        n_msgs: usize,
        add_ass: bool,
        buf: *mut c_char,
        buf_len: i32,
    ) -> i32;
}
