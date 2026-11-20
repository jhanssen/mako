// Hy4-flash against the reference implementation's logits (tests/data/hy4f_*_ref.bin, top-64 per position):
//   short: 15 prompt tokens one decode step each;
//   long:  2210 tokens of prose through prefill (past the 512-token window and, from 2048, the indexer's selection),
//          then its last 30 as decode steps after a prefill of the rest, against both the reference and the prefill;
//   a checkpoint taken after the short prompt and restored gives the same greedy continuation, and the long prompt's
//   context saved (checkpoint and positions), reset and loaded again does too; a prefill in other chunks gives the same
//   logits.
// With the vision projector, an image prompt (data/vision/shapes.png and a question about it; no reference logits for
// images): prefilled in chunks of 100 it gives the same logits as in chunks of 512, its last tokens as decode steps
// stay close to the prefill, a checkpoint after it repeats the greedy answer, and the answer names the colour asked.
//   hy4f_ref_test <model-00001.gguf> [mmproj.gguf]     (HY4F_IMAGE_ONLY=1: the image checks alone)
//   hy4f_ref_test - --import-ref <logits dir> short|long
// --import-ref turns a reference run's logits (logits.<pos>.f32) for a case's prompt (data/hy4f_<case>_ids.txt) into
// its stored top-k (data/hy4f_<case>_ref.bin). Exit 77: the model is missing (skipped).
#include "chat/template.hpp"
#include "hy4f/config.hpp"
#include "hy4f/engine.hpp"
#include "mtl/context.hpp"
#include "text/tokenizer.hpp"
#include "vision/encoder.hpp"
#include "vision/image.hpp"
#include "vision/mmproj.hpp"
#include "vision/prompt.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <cstring>
#include <format>
#include <fstream>
#include <iterator>
#include <numeric>
#include <print>
#include <span>
#include <sstream>
#include <string>
#include <vector>

namespace {

const std::filesystem::path kData = MAKO_TEST_DATA;
int failures = 0;

void check(bool ok, const std::string& what)
{
    std::println("  {}  {}", ok ? "ok  " : "FAIL", what);
    failures += !ok;
}

std::string read_file(const std::filesystem::path& p)
{
    std::ifstream f(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

std::vector<int32_t> read_ids(const std::string& name)
{
    std::vector<int32_t> ids;
    std::stringstream ss(read_file(kData / std::format("hy4f_{}_ids.txt", name)));
    for (std::string x; std::getline(ss, x, ',');)
        if (!x.empty() && x != "\n")
            ids.push_back(std::stoi(x));
    return ids;
}

std::vector<double> log_softmax(const float* x, size_t n)
{
    const float m = *std::max_element(x, x + n);
    double s = 0;
    for (size_t i = 0; i < n; ++i)
        s += std::exp(double(x[i] - m));
    const double lse = m + std::log(s);
    std::vector<double> y(n);
    for (size_t i = 0; i < n; ++i)
        y[i] = x[i] - lse;
    return y;
}

struct RefTopK {
    uint32_t positions = 0, k = 0;
    std::vector<float> lse;
    std::vector<std::vector<std::pair<int32_t, float>>> top;
};

RefTopK read_ref(const std::string& name)
{
    RefTopK r;
    const std::string raw = read_file(kData / std::format("hy4f_{}_ref.bin", name));
    if (raw.size() < 8)
        return r;
    const char* p = raw.data();
    std::memcpy(&r.positions, p, 4);
    std::memcpy(&r.k, p + 4, 4);
    p += 8;
    for (uint32_t i = 0; i < r.positions; ++i) {
        float lse;
        std::memcpy(&lse, p, 4);
        p += 4;
        r.lse.push_back(lse);
        auto& row = r.top.emplace_back();
        for (uint32_t j = 0; j < r.k; ++j, p += 8) {
            int32_t id;
            float v;
            std::memcpy(&id, p, 4);
            std::memcpy(&v, p + 4, 4);
            row.emplace_back(id, v);
        }
    }
    return r;
}

int import_ref(const std::filesystem::path& dir, const std::string& name)
{
    const uint32_t n = uint32_t(read_ids(name).size()), k = 64;
    std::ofstream out(kData / std::format("hy4f_{}_ref.bin", name), std::ios::binary);
    out.write(reinterpret_cast<const char*>(&n), 4);
    out.write(reinterpret_cast<const char*>(&k), 4);
    for (uint32_t pos = 0; pos < n; ++pos) {
        const std::string raw = read_file(dir / std::format("logits.{}.f32", pos));
        if (raw.empty()) {
            std::println(stderr, "missing reference logits at position {}", pos);
            return 1;
        }
        const auto* x = reinterpret_cast<const float*>(raw.data());
        const size_t nv = raw.size() / 4;
        const float lse = float(x[0] - log_softmax(x, nv)[0]);
        std::vector<int32_t> order(nv);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](int a, int b) { return x[a] > x[b]; });
        out.write(reinterpret_cast<const char*>(&lse), 4);
        for (uint32_t i = 0; i < k; ++i) {
            out.write(reinterpret_cast<const char*>(&order[i]), 4);
            out.write(reinterpret_cast<const char*>(&x[order[i]]), 4);
        }
    }
    std::println("wrote {} positions of top-{} reference logits to hy4f_{}_ref.bin", n, k, name);
    return 0;
}

// KL(ref || ours) over the reference's top-k and top-1 agreement, over positions [from, to).
struct Parity {
    const RefTopK& ref;
    uint32_t from = 0, to = UINT32_MAX;
    double kl_sum = 0, kl_max = 0, agree = 0;
    uint32_t n = 0;

    void add(uint32_t pos, const float* logits, size_t n_vocab)
    {
        if (pos < from || pos >= to || pos >= ref.positions)
            return;
        const auto ours = log_softmax(logits, n_vocab);
        double kl = 0;
        for (const auto& [id, v] : ref.top[pos]) {
            const double lp = v - ref.lse[pos];
            kl += std::exp(lp) * (lp - ours[id]);
        }
        kl_sum += kl;
        kl_max = std::max(kl_max, kl);
        agree += int(std::max_element(logits, logits + n_vocab) - logits) == ref.top[pos][0].first;
        ++n;
    }
    void check_limits(const std::string& what, double kl_limit, double top1_limit) const
    {
        const double kl = n ? kl_sum / n : 0, top1 = n ? agree / n : 0;
        check(n > 0 && kl < kl_limit && top1 > top1_limit,
              std::format("{} ({} positions): KL mean {:.5f} (limit {}), max {:.4f}, top-1 {:.2f}% (limit {}%)", what,
                          n, kl, kl_limit, kl_max, 100 * top1, 100 * top1_limit));
    }
};

double kl_full(const std::vector<float>& a, const std::vector<float>& b)
{
    const auto la = log_softmax(a.data(), a.size()), lb = log_softmax(b.data(), b.size());
    double kl = 0;
    for (size_t i = 0; i < a.size(); ++i)
        kl += std::exp(la[i]) * (la[i] - lb[i]);
    return kl;
}

int32_t argmax(const std::vector<float>& l) { return int32_t(std::max_element(l.begin(), l.end()) - l.begin()); }

void image_checks(const mako::gguf::ModelFiles& model, mako::hy4f::Engine& engine, const char* mmproj_path,
                  const mako::mtl::Context& ctx)
{
    namespace vision = mako::vision;
    auto mm = vision::open_mmproj(mmproj_path);
    auto tok = mako::text::Tokenizer::load(model.meta());
    if (!mm || !tok) {
        check(false, std::format("image: {}", !mm ? mm.error() : tok.error()));
        return;
    }
    const std::string png = read_file(kData / "vision" / "shapes.png");
    const auto prep = vision::prepare(mm->prep, vision::decode_image(std::as_bytes(std::span(png.data(), png.size()))));
    vision::Encoder enc(ctx, *mm);
    const auto img = enc.encode(prep);
    const mako::chat::Template tmpl{std::string(model.meta().get_string("tokenizer.chat_template").value_or(""))};
    const auto rendered = tmpl.render(
        R"({"messages":[{"role":"user","content":[{"type":"image"},{"type":"text","text":"What colour is the circle? Answer in one word."}]}],"add_generation_prompt":true,"reasoning_effort":"no_think"})");
    const auto ids = tok->encode(rendered.text, rendered.input);
    const uint64_t key = 1;
    const uint32_t cols = vision::Encoder::cols_for(*mm, prep);
    auto prompt = vision::expand_images(ids, tok->find(vision::placeholder(mm->prep.kind)), std::span(&img.rows, 1),
                                        std::span(&key, 1), std::span(&cols, 1));
    std::vector<mako::gen::InputRows> rows{prompt.rows[0].rows};
    rows[0].rows = img.embd.data();
    const auto& toks = prompt.tokens;
    const uint32_t n = uint32_t(toks.size()), V = engine.n_vocab(), image_end = rows[0].pos + rows[0].count;
    std::println("  image: {} x {} cells, prompt {} tokens (rows {}..{})", img.cells_x, img.cells_y, n, rows[0].pos,
                 image_end - 1);

    // The image's last rows and the text after it.
    const uint32_t from = image_end - 16;
    std::vector<std::vector<float>> l512(n - from);
    engine.reset();
    engine.set_input_rows(rows);
    engine.set_logits_sink([&](uint32_t pos, const float* l) {
        if (pos >= from)
            l512[pos - from].assign(l, l + V);
    });
    engine.prefill(toks);
    engine.set_logits_sink({});
    const auto cp = engine.checkpoint();
    const auto saved = engine.save_positions(cp.pos);
    const std::vector<float> first = engine.logits();
    // Up to the end of the answer.
    auto greedy = [&](int count) {
        std::vector<int32_t> out{argmax(first)};
        for (int i = 1; i < count && out.back() != tok->eos(); ++i)
            out.push_back(argmax(engine.step(out.back())));
        return out;
    };
    const auto a = greedy(16);
    const std::string answer = tok->decode(a, false);
    std::string lower;
    for (char c : answer)
        lower += char(std::tolower(static_cast<unsigned char>(c)));
    check(lower.find("red") != std::string::npos && a.back() == tok->eos(),
          std::format("image: the answer names the circle's colour: {:?} ({} tokens to the end)", answer, a.size()));
    engine.reset();
    engine.load_positions(cp.pos, saved);
    engine.restore(cp);
    check(greedy(16) == a, "image: the greedy answer is the same after the context is saved, reset and loaded");

    engine.reset();
    engine.set_prefill_chunk(100);
    bool same = true;
    engine.set_logits_sink([&](uint32_t pos, const float* l) {
        if (pos >= from)
            same = same && std::equal(l, l + V, l512[pos - from].begin());
    });
    engine.prefill(toks);
    engine.set_logits_sink({});
    engine.set_prefill_chunk(512);
    check(same, "image: a prefill in chunks of 100 gives the same logits as in chunks of 512 (the image's last rows on)");

    // The last 4 image rows and the text after as decode steps, compared where the logits are used: from the image's
    // end token on (an image row's next-token logits are diffuse: up to KL 0.5 between the two paths).
    const uint32_t steps_from = image_end - 4;
    engine.reset();
    engine.prefill(std::span(toks).first(steps_from));
    double kl = 0, kl_max = 0;
    for (uint32_t p = steps_from; p < n; ++p) {
        const auto& l = engine.step(toks[p]);
        const double k = kl_full(l512[p - from], std::vector<float>(l.begin(), l.end()));
        if (p < image_end)
            continue;
        kl += k;
        kl_max = std::max(kl_max, k);
    }
    kl /= n - image_end;
    // Measured: 2.7e-3 (text alone, the long prompt's tail: 1.8e-3).
    check(kl < 4e-3, std::format("image: decode steps from the image's last rows vs the prefill, from its end on: KL mean {:.2e} (limit 4e-3), max {:.2e}", kl, kl_max));
    engine.set_input_rows({});
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::println(stderr, "usage: hy4f_ref_test <model-00001.gguf> | - --import-ref <dir> short|long");
        return 2;
    }
    if (argc == 5 && std::string_view(argv[2]) == "--import-ref")
        return import_ref(argv[3], argv[4]);
    if (!std::filesystem::exists(argv[1])) {
        std::println("model {} not found: skipped", argv[1]);
        return 77;
    }
    auto m = mako::gguf::ModelFiles::open(argv[1]);
    if (!m) {
        std::println(stderr, "{}", m.error());
        return 1;
    }
    auto cfg = mako::hy4f::Config::load(*m);
    if (!cfg) {
        std::println(stderr, "{}", cfg.error());
        return 1;
    }
    mako::mtl::Context ctx;
    mako::hy4f::Options opts;
    opts.max_ctx = 4096;
    mako::hy4f::Engine engine(*m, *cfg, opts, ctx);
    const size_t V = cfg->n_vocab;

    // Limits: ~1.1x the KL measured when set, top-1 about a point below. Measured: short decode KL 0.0155, top-1
    // 93.3% (14 of 15); long prefill 0.0195, 92.9%, of it 512-2047 0.0194, 92.8%, 2048 on 0.0131, 93.8%; the tail's
    // steps 0.0102, 96.7%, against its prefill KL 1.77e-3.
    if (!std::getenv("HY4F_IMAGE_ONLY")) {
    {
        const auto ids = read_ids("short");
        const RefTopK ref = read_ref("short");
        Parity dec{ref};
        for (uint32_t p = 0; p < ids.size(); ++p)
            dec.add(p, engine.step(ids[p]).data(), V);
        dec.check_limits("short: decode vs reference", 0.017, 0.9);

        // A checkpoint here, then 8 greedy tokens; back to it, the same 8.
        const auto cp = engine.checkpoint();
        const std::vector<float> first = engine.logits();
        auto greedy = [&] {
            std::vector<int32_t> out{argmax(first)};
            for (int i = 1; i < 8; ++i)
                out.push_back(argmax(engine.step(out.back())));
            return out;
        };
        const auto a = greedy();
        engine.restore(cp);
        const auto b = greedy();
        check(a == b, "short: the greedy continuation is the same after a checkpoint and restore");
    }
    {
        engine.reset();
        const auto ids = read_ids("long");
        const RefTopK ref = read_ref("long");
        Parity all{ref}, window{ref, 512, 2048}, select{ref, 2048};
        constexpr uint32_t kTail = 30;
        const uint32_t tail0 = uint32_t(ids.size()) - kTail;
        std::vector<std::vector<float>> tail(kTail);
        engine.set_logits_sink([&](uint32_t pos, const float* l) {
            all.add(pos, l, V);
            window.add(pos, l, V);
            select.add(pos, l, V);
            if (pos >= tail0)
                tail[pos - tail0].assign(l, l + V);
        });
        engine.prefill(ids);
        engine.set_logits_sink({});
        // The context carried out (a checkpoint and the positional caches), 8 greedy tokens; back in after a reset,
        // the same 8 (the window layers' sliding buffers have moved several times by now).
        {
            const auto cp = engine.checkpoint();
            const auto saved = engine.save_positions(cp.pos);
            const std::vector<float> first = engine.logits();
            auto greedy = [&] {
                std::vector<int32_t> out{argmax(first)};
                for (int i = 1; i < 8; ++i)
                    out.push_back(argmax(engine.step(out.back())));
                return out;
            };
            const auto a = greedy();
            engine.reset();
            engine.load_positions(cp.pos, saved);
            engine.restore(cp);
            const auto b = greedy();
            check(a == b, "long: the greedy continuation is the same after the context is saved, reset and loaded");
        }
        all.check_limits("long: prefill vs reference", 0.0215, 0.918);
        window.check_limits("long: prefill vs reference, positions 512-2047", 0.0215, 0.918);
        select.check_limits("long: prefill vs reference, positions 2048 on (selecting)", 0.0145, 0.928);

        // Chunks of 100: every row computed exactly as with the default chunk (the window layers' keys and the
        // sparse sums do not depend on the grouping).
        {
            engine.reset();
            engine.set_prefill_chunk(100);
            bool same = true;
            engine.set_logits_sink([&](uint32_t pos, const float* l) {
                if (pos >= tail0)
                    same = same && std::equal(l, l + V, tail[pos - tail0].begin());
            });
            engine.prefill(ids);
            engine.set_logits_sink({});
            engine.set_prefill_chunk(opts.prefill_chunk);
            check(same, "long: a prefill in chunks of 100 gives the same logits as in chunks of 512 (the tail's)");
        }
        engine.reset();
        engine.prefill(std::span(ids).first(tail0));
        Parity steps{ref, tail0};
        double kl = 0, kl_max = 0;
        for (uint32_t p = tail0; p < ids.size(); ++p) {
            const auto& l = engine.step(ids[p]);
            steps.add(p, l.data(), V);
            const double k = kl_full(tail[p - tail0], std::vector<float>(l.begin(), l.end()));
            kl += k;
            kl_max = std::max(kl_max, k);
        }
        steps.check_limits("long: the tail's decode steps vs reference", 0.0112, 0.93);
        check(kl / kTail < 1.95e-3, std::format("long: the tail's decode steps vs its prefill: KL mean {:.2e} (limit 1.95e-3), max {:.2e}",
                                           kl / kTail, kl_max));
    }
    }
    if (argc >= 3 && std::string_view(argv[2]).size() && std::filesystem::exists(argv[2]))
        image_checks(*m, engine, argv[2], ctx);
    else
        std::println("  (no vision projector: the image checks skipped)");
    std::println("{}", failures ? std::format("{} failures", failures) : std::string("all passed"));
    return failures ? 1 : 0;
}
