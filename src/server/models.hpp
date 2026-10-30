#pragma once

#include "core/engine_base.hpp"
#include "gen/model.hpp"
#include "gen/pool.hpp"
#include "image/engine.hpp"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

namespace mako::server {

// The models a server runs, one resident at a time: the LLM, and an image model when one is given. A request leases
// the one it needs. A lease of the image model waits until no LLM lease is out (LLM leases asked for meanwhile wait
// for it), then parks the conversations (gen::SessionPool::suspend), unloads the LLM and loads the image model; the
// next LLM lease loads the LLM again and gives the pool its conversations back.
class Models {
public:
    struct Llm {
        std::unique_ptr<gen::Model> model;
        core::EngineBase* base = nullptr; // the same engine, for its expert cache
    };
    using LoadLlm = std::function<Llm()>;
    using LoadImage = std::function<std::unique_ptr<image::Engine>()>;

    // Loads the LLM; load_image may be empty (no image model).
    Models(LoadLlm load_llm, LoadImage load_image);
    // An LLM the caller owns, never unloaded, and no image model.
    explicit Models(gen::Model& fixed);
    ~Models();
    Models(const Models&) = delete;
    Models& operator=(const Models&) = delete;

    // The LLM as first loaded (a reload is the same model with the same options): for the pool and fixed figures.
    gen::Model& first_llm() { return *llm_model_; }
    // The pool whose conversations live across swaps, and the identity they are kept under.
    void attach(gen::SessionPool& pool, std::string identity);
    bool has_image() const { return bool(load_image_); }

    template <typename T> class Lease {
    public:
        Lease(Models& m, T& v, void (Models::*release)()) : m_(&m), v_(&v), release_(release) {}
        Lease(Lease&& o) noexcept : m_(std::exchange(o.m_, nullptr)), v_(o.v_), release_(o.release_) {}
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        ~Lease()
        {
            if (m_)
                (m_->*release_)();
        }
        T& operator*() const { return *v_; }
        T* operator->() const { return v_; }

    private:
        Models* m_;
        T* v_;
        void (Models::*release_)();
    };
    Lease<gen::Model> llm();
    // Throws std::runtime_error without an image model.
    Lease<image::Engine> image();

    // f(the LLM's engine) while the LLM is loaded; false (f not called) otherwise. Any thread.
    bool with_llm(const std::function<void(core::EngineBase&)>& f);
    // "llm", "image" or "none" (between an unload and a load).
    const char* resident();

private:
    void release_llm();
    void release_image();

    LoadLlm load_llm_;
    LoadImage load_image_;
    Llm llm_;
    gen::Model* llm_model_ = nullptr; // llm_.model's, or the fixed one; null while unloaded
    std::unique_ptr<image::Engine> image_;
    gen::SessionPool* pool_ = nullptr;
    std::string identity_;
    std::mutex mutex_; // the counts and what is loaded; held through a swap
    std::condition_variable cv_;
    uint32_t llm_out_ = 0, image_waiting_ = 0;
    bool image_out_ = false;
    std::mutex llm_mutex_; // llm_.base for with_llm(), apart from the swap's long hold
    std::atomic<const char*> resident_{"llm"};
};

} // namespace mako::server
