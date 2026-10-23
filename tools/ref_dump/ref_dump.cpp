// Runs token ids one at a time through the reference implementation on the CPU and writes the named
// intermediate tensors of every step as raw fp32: <out>/<name>.<pos>.f32 (name as "<tensor>-<layer>").
//   ref_dump <model.gguf> <id,id,...> <out_dir> [name-substring ...]
// An id list starting with '@' is text to tokenize instead; the ids are printed.
// REF_INDEXER_TOP_K=N in the environment overrides the model's sparse-attention selection width.
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <print>
#include <sstream>
#include <string>
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
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t))
        return true;
    d.buf.resize(ggml_nelements(t));
    ggml_backend_tensor_get(t, d.buf.data(), 0, ggml_nbytes(t));
    std::ofstream out(d.dir / (std::string(t->name) + "." + std::to_string(d.pos) + ".f32"), std::ios::binary);
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
    cp.n_ctx = 512;
    cp.n_batch = 1;
    cp.n_ubatch = 1;
    cp.cb_eval = on_tensor;
    cp.cb_eval_user_data = &d;
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
