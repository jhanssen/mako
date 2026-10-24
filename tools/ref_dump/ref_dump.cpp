// Runs token ids one at a time through the reference implementation on the CPU and writes the named
// intermediate tensors of every step as raw fp32 or int32: <out>/<name>.<pos>.f32|.i32 (name as "<tensor>-<layer>").
//   ref_dump <model.gguf> <id,id,...> <out_dir> [name-substring ...]
// An id list starting with '@' is text to tokenize instead; the ids are printed.
// REF_INDEXER_TOP_K=N in the environment overrides the model's sparse-attention selection width.
// REF_BATCH=N: logits only (no intermediate tensors), the ids fed N at a time: much faster for long prompts.
// REF_TOKENIZE=<file>: load only the vocabulary, tokenize each 0x1E-separated case of <file> (special tokens
// parsed) and print one line of ids per case; the other arguments are ignored.
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <print>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Dump {
    std::filesystem::path dir;
    std::vector<std::string> filters;
    int pos = 0;
    std::vector<float> buf;
};

bool wanted(const Dump& d, const char* name)
{
    if (d.filters.empty())
        return true;
    for (const auto& f : d.filters)
        if (std::string_view(name).find(f) != std::string_view::npos)
            return true;
    return false;
}

bool on_tensor(ggml_tensor* t, bool ask, void* user)
{
    auto& d = *static_cast<Dump*>(user);
    if (ask)
        return wanted(d, t->name);
    if ((t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_I32) || !ggml_is_contiguous(t))
        return true;
    d.buf.resize(ggml_nelements(t));
    ggml_backend_tensor_get(t, d.buf.data(), 0, ggml_nbytes(t));
    const char* ext = t->type == GGML_TYPE_F32 ? ".f32" : ".i32";
    std::ofstream out(d.dir / (std::string(t->name) + "." + std::to_string(d.pos) + ext), std::ios::binary);
    out.write(reinterpret_cast<const char*>(d.buf.data()), std::streamsize(d.buf.size() * 4));
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::println(stderr, "usage: ref_dump <model.gguf> <id,id,...> <out_dir> [name-substring ...]");
        return 2;
    }
    Dump d;
    d.dir = argv[3];
    std::filesystem::create_directories(d.dir);
    for (int i = 4; i < argc; ++i)
        d.filters.emplace_back(argv[i]);
    std::vector<llama_token> ids;
    const std::string id_arg = argv[2];
    if (id_arg.empty() || id_arg[0] != '@') {
        std::stringstream ss(id_arg);
        for (std::string tok; std::getline(ss, tok, ',');)
            if (!tok.empty())
                ids.push_back(std::stoi(tok));
    }

    llama_backend_init();
    if (const char* tf = std::getenv("REF_TOKENIZE")) {
        auto vp = llama_model_default_params();
        vp.vocab_only = true;
        llama_model* vm = llama_model_load_from_file(argv[1], vp);
        if (!vm)
            return 1;
        const llama_vocab* v = llama_model_get_vocab(vm);
        std::ifstream in(tf, std::ios::binary);
        std::string all((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        std::stringstream cases(all);
        for (std::string c; std::getline(cases, c, '\x1e');) {
            std::vector<llama_token> t(c.size() + 16);
            const int n = llama_tokenize(v, c.data(), int32_t(c.size()), t.data(), int32_t(t.size()), false, true);
            std::string line;
            for (int i = 0; i < n; ++i)
                line += (i ? "," : "") + std::to_string(t[i]);
            std::println("{}", line);
        }
        llama_model_free(vm);
        return 0;
    }
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    std::vector<llama_model_kv_override> overrides;
    if (const char* k = std::getenv("REF_INDEXER_TOP_K")) {
        llama_model_kv_override o{};
        o.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        std::strncpy(o.key, "glm5-next.attention.indexer.top_k", sizeof(o.key) - 1);
        o.val_i64 = std::atoll(k);
        overrides.push_back(o);
        overrides.push_back(llama_model_kv_override{});
        mp.kv_overrides = overrides.data();
    }
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model)
        return 1;
    auto cp = llama_context_default_params();
    cp.n_ctx = std::max<uint32_t>(512, uint32_t(ids.size() + id_arg.size() / 2 + 16));
    const int batch = std::getenv("REF_BATCH") ? std::max(1, std::atoi(std::getenv("REF_BATCH"))) : 0;
    cp.n_batch = batch ? uint32_t(batch) : 1;
    cp.n_ubatch = cp.n_batch;
    if (batch) {
        const int threads = int(std::thread::hardware_concurrency());
        cp.n_threads = threads;
        cp.n_threads_batch = threads;
    } else {
        cp.cb_eval = on_tensor;
        cp.cb_eval_user_data = &d;
    }
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx)
        return 1;

    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    if (!id_arg.empty() && id_arg[0] == '@') {
        const std::string text = id_arg.substr(1);
        ids.resize(text.size() + 8);
        const int n = llama_tokenize(vocab, text.data(), int32_t(text.size()), ids.data(), int32_t(ids.size()), false, true);
        if (n < 0)
            return 1;
        ids.resize(n);
        std::string list;
        for (auto id : ids)
            list += std::to_string(id) + ",";
        std::println("ids {}", list);
    }
    if (batch) {
        llama_batch b = llama_batch_init(batch, 0, 1);
        for (size_t i0 = 0; i0 < ids.size(); i0 += size_t(batch)) {
            const int n = int(std::min<size_t>(size_t(batch), ids.size() - i0));
            b.n_tokens = n;
            for (int j = 0; j < n; ++j) {
                b.token[j] = ids[i0 + j];
                b.pos[j] = llama_pos(i0 + j);
                b.n_seq_id[j] = 1;
                b.seq_id[j][0] = 0;
                b.logits[j] = 1;
            }
            if (llama_decode(ctx, b) != 0) {
                std::println(stderr, "decode failed at {}", i0);
                return 1;
            }
            for (int j = 0; j < n; ++j) {
                const float* logits = llama_get_logits_ith(ctx, j);
                std::ofstream out(d.dir / ("logits." + std::to_string(i0 + j) + ".f32"), std::ios::binary);
                out.write(reinterpret_cast<const char*>(logits), std::streamsize(n_vocab) * 4);
            }
            std::println("positions {}..{} done", i0, i0 + n - 1);
        }
        llama_batch_free(b);
        llama_free(ctx);
        llama_model_free(model);
        return 0;
    }
    for (size_t i = 0; i < ids.size(); ++i) {
        d.pos = int(i);
        llama_token tok = ids[i];
        if (llama_decode(ctx, llama_batch_get_one(&tok, 1)) != 0) {
            std::println(stderr, "decode failed at {}", i);
            return 1;
        }
        const float* logits = llama_get_logits_ith(ctx, -1);
        std::ofstream out(d.dir / ("logits." + std::to_string(i) + ".f32"), std::ios::binary);
        out.write(reinterpret_cast<const char*>(logits), std::streamsize(n_vocab) * 4);
        int best = 0;
        for (int v = 1; v < n_vocab; ++v)
            if (logits[v] > logits[best])
                best = v;
        std::println("pos {} in {} argmax {} ({:.3f})", i, tok, best, logits[best]);
    }
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
