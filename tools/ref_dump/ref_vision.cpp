// Runs an image through the reference implementation's multimodal path (preprocessing, vision encoder, projector)
// and writes each image chunk's embeddings as raw fp32 rows of the projector's width: <out>.<chunk>.f32. Prints the
// prompt's chunk layout (text ids, image token counts).
//   ref_vision <text model.gguf | -> <mmproj.gguf> <image> <out> [gpu]
// The text model is loaded vocabulary-only (for the text around the image); "-" runs without one, for projectors
// that add no vocabulary tokens around an image. gpu: 1 runs the encoder on the GPU (default: CPU).
// REF_VISION_DUMP=<name,name,...>: the encoder's tensors of those names (e.g. inp_raw, patch_bias, vit_out,
// layer_out-0) also go to <out>.<name>.f32, contiguous fp32, first dimension fastest; a name seen again gets
// a .2, .3, ... suffix. REF_VISION_STOP=<name>: exit once that tensor is written.
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <map>
#include <print>
#include <sstream>
#include <string>
#include <vector>

namespace {

// clip.vision.projection_dim of the mmproj: the width of an image token's embedding.
int projection_dim(const char* path)
{
    gguf_init_params p{.no_alloc = true, .ctx = nullptr};
    gguf_context* g = gguf_init_from_file(path, p);
    if (!g)
        return 0;
    const int64_t k = gguf_find_key(g, "clip.vision.projection_dim");
    const int n = k >= 0 ? int(gguf_get_val_u32(g, k)) : 0;
    gguf_free(g);
    return n;
}

struct Dump {
    std::string out;
    std::vector<std::string> names;
    std::map<std::string, int> seen;
};

float element(const ggml_tensor* t, int64_t i0, int64_t i1, int64_t i2, int64_t i3)
{
    const char* p = static_cast<const char*>(t->data) + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
    switch (t->type) {
    case GGML_TYPE_F32: return *reinterpret_cast<const float*>(p);
    case GGML_TYPE_F16: return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(p));
    case GGML_TYPE_BF16: return ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t*>(p));
    case GGML_TYPE_I32: return float(*reinterpret_cast<const int32_t*>(p));
    default: return 0.0f;
    }
}

void write_tensor(Dump& d, const ggml_tensor* t)
{
    const std::string name = ggml_get_name(t);
    const int k = ++d.seen[name];
    const std::string path = std::format("{}.{}{}.f32", d.out, name, k > 1 ? std::format(".{}", k) : "");
    std::vector<float> v;
    v.reserve(size_t(ggml_nelements(t)));
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1)
                for (int64_t i0 = 0; i0 < t->ne[0]; ++i0)
                    v.push_back(element(t, i0, i1, i2, i3));
    if (std::FILE* f = std::fopen(path.c_str(), "wb")) {
        std::fwrite(v.data(), 4, v.size(), f);
        std::fclose(f);
    }
    std::println("{} [{}, {}, {}, {}] {} -> {}", name, t->ne[0], t->ne[1], t->ne[2], t->ne[3], ggml_type_name(t->type),
                 path);
    if (const char* stop = std::getenv("REF_VISION_STOP"); stop && name == stop) {
        std::fflush(stdout);
        std::_Exit(0);
    }
}

// Writes the named tensors once computed (CPU backend: t->data is host memory), honouring their strides. Graph
// inputs (inp_raw, positions) are never computed: they are written when the first node reading them runs.
bool dump_tensor(ggml_tensor* t, bool ask, void* user)
{
    auto& d = *static_cast<Dump*>(user);
    const auto wanted = [&](const ggml_tensor* x) {
        if (!x)
            return false;
        for (const auto& n : d.names)
            if (n == ggml_get_name(x))
                return true;
        return false;
    };
    const auto pending_input = [&](const ggml_tensor* x) {
        return wanted(x) && x->op == GGML_OP_NONE && !d.seen.contains(ggml_get_name(x));
    };
    bool any = wanted(t);
    for (const ggml_tensor* s : t->src)
        any |= pending_input(s);
    if (ask || !any)
        return any;
    for (const ggml_tensor* s : t->src)
        if (pending_input(s))
            write_tensor(d, s);
    if (wanted(t))
        write_tensor(d, t);
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 5) {
        std::println(stderr, "usage: ref_vision <text model.gguf | -> <mmproj.gguf> <image> <out> [gpu]");
        return 2;
    }
    const bool with_text = std::strcmp(argv[1], "-") != 0;
    const int n_embd = projection_dim(argv[2]);
    if (n_embd <= 0) {
        std::println(stderr, "{}: no clip.vision.projection_dim", argv[2]);
        return 1;
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model* model = with_text ? llama_model_load_from_file(argv[1], mp) : nullptr;
    if (with_text && !model)
        return 1;
    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = argc > 5 && std::atoi(argv[5]) != 0;
    cp.n_threads = 16;
    Dump dump{argv[4], {}, {}};
    if (const char* names = std::getenv("REF_VISION_DUMP")) {
        std::stringstream ss(names);
        for (std::string n; std::getline(ss, n, ',');)
            if (!n.empty())
                dump.names.push_back(n);
        cp.cb_eval = dump_tensor;
        cp.cb_eval_user_data = &dump;
    }
    mtmd_context* ctx = mtmd_init_from_file(argv[2], model, cp);
    if (!ctx)
        return 1;
    const mtmd_helper_bitmap_wrapper bw =
        mtmd_helper_bitmap_init_from_file(ctx, argv[3], false, mtmd_helper_init_opt_default());
    if (!bw.bitmap)
        return 1;
    std::println("image {} x {}", mtmd_bitmap_get_nx(bw.bitmap), mtmd_bitmap_get_ny(bw.bitmap));
    const std::string text = with_text ? std::format("A{}B", mtmd_default_marker()) : mtmd_default_marker();
    const mtmd_input_text in{text.c_str(), text.size(), false, true};
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    const mtmd_bitmap* bitmaps[1] = {bw.bitmap};
    if (const int32_t r = mtmd_tokenize(ctx, chunks, &in, bitmaps, 1); r != 0) {
        std::println(stderr, "tokenize failed: {}", r);
        return 1;
    }
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
        const mtmd_input_chunk* c = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n = 0;
            const llama_token* t = mtmd_input_chunk_get_tokens_text(c, &n);
            std::print("chunk {} text:", i);
            for (size_t k = 0; k < n; ++k)
                std::print(" {}", t[k]);
            std::println("");
            continue;
        }
        const size_t n = mtmd_input_chunk_get_n_tokens(c);
        if (const int32_t r = mtmd_encode_chunk(ctx, c); r != 0) {
            std::println(stderr, "encode failed: {}", r);
            return 1;
        }
        const float* e = mtmd_get_output_embd(ctx);
        const std::string path = std::format("{}.{}.f32", argv[4], i);
        std::FILE* out = std::fopen(path.c_str(), "wb");
        if (!out || std::fwrite(e, sizeof(float), n * size_t(n_embd), out) != n * size_t(n_embd)) {
            std::println(stderr, "cannot write {}", path);
            return 1;
        }
        std::fclose(out);
        std::println("chunk {} image: {} tokens x {}, {} positions -> {}", i, n, n_embd,
                     mtmd_input_chunk_get_n_pos(c), path);
    }
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bw.bitmap);
    mtmd_free(ctx);
    if (model)
        llama_model_free(model);
    return 0;
}
