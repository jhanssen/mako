// Runs an image through the reference implementation's multimodal path (preprocessing, vision encoder, projector)
// and writes each image chunk's embeddings as raw fp32 rows of the projector's width: <out>.<chunk>.f32. Prints the
// prompt's chunk layout (text ids, image token counts).
//   ref_vision <text model.gguf | -> <mmproj.gguf> <image> <out> [gpu]
// The text model is loaded vocabulary-only (for the text around the image); "-" runs without one, for projectors
// that add no vocabulary tokens around an image. gpu: 1 runs the encoder on the GPU (default: CPU).
#include "gguf.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <print>
#include <string>

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
