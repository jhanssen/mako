"""Writes a vocabulary-only GGUF (architecture "llama", dummy geometry) carrying a model's tokenizer.ggml.* keys, so the
reference tokenizer can tokenize for an architecture it does not load (ref_dump REF_TOKENIZE=<cases>).

    python3 tools/vocab_only_gguf.py <model shard 1.gguf> <out.gguf> <llama.cpp>/gguf-py

Needs numpy and pyyaml (gguf-py's imports).
"""
import sys
sys.path.insert(0, sys.argv[3])
import gguf
r = gguf.GGUFReader(sys.argv[1])
w = gguf.GGUFWriter(sys.argv[2], "llama")
w.add_context_length(4096); w.add_embedding_length(64); w.add_block_count(1); w.add_feed_forward_length(64)
w.add_head_count(1); w.add_layer_norm_rms_eps(1e-5)
for name, f in r.fields.items():
    if not name.startswith("tokenizer.ggml"):
        continue
    v = f.contents()
    if name == "tokenizer.ggml.model": w.add_tokenizer_model(v)
    elif name == "tokenizer.ggml.pre": w.add_tokenizer_pre(v)
    elif name == "tokenizer.ggml.tokens": w.add_token_list(v)
    elif name == "tokenizer.ggml.token_type": w.add_token_types(v)
    elif name == "tokenizer.ggml.merges": w.add_token_merges(v)
    elif name == "tokenizer.ggml.bos_token_id": w.add_bos_token_id(v)
    elif name == "tokenizer.ggml.eos_token_id": w.add_eos_token_id(v)
    elif name == "tokenizer.ggml.padding_token_id": w.add_pad_token_id(v)
    elif name == "tokenizer.ggml.add_bos_token": w.add_add_bos_token(v)
    elif name == "tokenizer.ggml.add_eos_token": w.add_add_eos_token(v)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
