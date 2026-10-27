// One question about an image, answered greedily: the chat template with an image part, the image through the
// projector, the answer printed (reasoning included).
//   vision_chat <model.gguf> <mmproj.gguf> <image> <question> [-n N] [--ctx N] [--no-think 1] [--image-max-tokens N]
#include "chat/template.hpp"
#include "dsv41/config.hpp"
#include "dsv41/engine.hpp"
#include "dsv41/engram.hpp"
#include "gen/session.hpp"
#include "glm/config.hpp"
#include "glm/engine.hpp"
#include "mtl/context.hpp"
#include "text/tokenizer.hpp"
#include "vision/encoder.hpp"
#include "vision/image.hpp"
#include "vision/mmproj.hpp"
#include "vision/prompt.hpp"

#include <chrono>
#include <format>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <print>
#include <string>
#include <vector>

namespace {

double ms_since(std::chrono::steady_clock::time_point t)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

std::string json_string(std::string_view s)
{
    std::string out = "\"";
    for (const char c : s) {
        if (c == '"' || c == '\\')
            out += '\\';
        if (c == '\n')
            out += "\\n";
        else
            out += c;
    }
    return out + "\"";
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 5) {
        std::println(stderr, "usage: vision_chat <model.gguf> <mmproj.gguf> <image> <question> [-n N] [--ctx N] [--no-think 1]");
        return 2;
    }
    uint32_t n_gen = 512, max_ctx = 16384;
    bool no_think = false;
    uint32_t image_max_tokens = 0;
    for (int i = 5; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "-n")
            n_gen = uint32_t(std::stoul(argv[i + 1]));
        else if (a == "--ctx")
            max_ctx = uint32_t(std::stoul(argv[i + 1]));
        else if (a == "--no-think")
            no_think = std::string_view(argv[i + 1]) != "0";
        else if (a == "--image-max-tokens")
            image_max_tokens = uint32_t(std::stoul(argv[i + 1]));
        else {
            std::println(stderr, "unknown option {}", a);
            return 2;
        }
    }
    try {
        auto model = mako::gguf::ModelFiles::open(argv[1]);
        if (!model)
            throw std::runtime_error(model.error());
        const bool deepseek = model->meta().get_string("general.architecture").value_or("") == "deepseek41";
        if (deepseek)
            if (auto r = mako::dsv41::Engram::add_named(*model, argv[1]); !r)
                throw std::runtime_error(r.error());
        const auto& meta = model->meta();
        auto tok = mako::text::Tokenizer::load(meta);
        if (!tok)
            throw std::runtime_error(tok.error());
        const mako::chat::Template tmpl{std::string(meta.get_string("tokenizer.chat_template").value_or(""))};

        auto mm = mako::vision::open_mmproj(argv[2]);
        if (!mm)
            throw std::runtime_error(mm.error());
        mm->prep.max_tokens = image_max_tokens;
        std::ifstream in(argv[3], std::ios::binary);
        const std::vector<char> bytes{std::istreambuf_iterator<char>(in), {}};
        const auto img = mako::vision::decode_image(std::as_bytes(std::span(bytes)));

        mako::mtl::Context ctx;
        std::unique_ptr<mako::glm::Engine> glm;
        std::unique_ptr<mako::dsv41::Engine> ds;
        mako::gen::Model* engine = nullptr;
        if (deepseek) {
            auto cfg = mako::dsv41::Config::load(*model);
            if (!cfg)
                throw std::runtime_error(cfg.error());
            mako::dsv41::Options opts;
            opts.max_ctx = max_ctx;
            ds = std::make_unique<mako::dsv41::Engine>(*model, *cfg, opts, ctx);
            engine = ds.get();
        } else {
            auto cfg = mako::glm::Config::load(*model);
            if (!cfg)
                throw std::runtime_error(cfg.error());
            mako::glm::Options opts;
            opts.max_ctx = max_ctx;
            glm = std::make_unique<mako::glm::Engine>(*model, *cfg, opts, ctx);
            engine = glm.get();
        }

        auto t0 = std::chrono::steady_clock::now();
        mako::vision::Encoder enc(ctx, *mm);
        const auto prep = mako::vision::prepare(mm->prep, img);
        const mako::vision::Encoder::Output rows = enc.encode(prep);
        std::println("image {} x {}: {} rows ({} x {} cells) in {:.0f} ms", img.width, img.height, rows.rows,
                     rows.cells_x, rows.cells_y, ms_since(t0));

        std::string context = std::format(
            R"({{"messages":[{{"role":"user","content":[{{"type":"image"}},{{"type":"text","text":{}}}]}}],"add_generation_prompt":true{}}})",
            json_string(argv[4]), no_think ? R"(,"enable_thinking":false)" : "");
        const auto rendered = tmpl.render(context);
        const auto ids = tok->encode(rendered.text, rendered.input);
        const int32_t ph = tok->find(mako::vision::placeholder(mm->prep.kind));
        if (ph < 0)
            throw std::runtime_error("the model's vocabulary has no image placeholder token");
        const uint64_t key = std::hash<std::string_view>{}(std::string_view(bytes.data(), bytes.size()));
        auto prompt = mako::vision::expand_images(ids, ph, std::span(&rows.rows, 1), std::span(&key, 1));
        prompt.rows[0].rows.rows = rows.embd.data();
        std::println("prompt: {} tokens ({} of them image rows)", prompt.tokens.size(), rows.rows);

        mako::gen::Session session(*engine);
        mako::gen::GenerateParams gp;
        gp.sampling.temperature = 0;
        gp.max_tokens = n_gen;
        gp.speculative = false;
        gp.stop_tokens = {tok->eos(), tok->eot()};
        gp.input_rows = prompt.rows;
        std::vector<int32_t> out;
        const auto r = session.generate(prompt.tokens, gp, [&](int32_t t) {
            out.push_back(t);
            return true;
        });
        std::println("prefill {:.0f} ms, {} tokens in {:.0f} ms\n---\n{}", r.prefill_ms, r.tokens.size(), r.decode_ms,
                     tok->decode(out, false));
    } catch (const std::exception& e) {
        std::println(stderr, "{}", e.what());
        return 1;
    }
    return 0;
}
