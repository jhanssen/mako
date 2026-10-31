#pragma once

#include "core/engine_base.hpp"
#include "gen/model.hpp"
#include "gen/pool.hpp"
#include "memory/lender.hpp"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

namespace mako::server {

// The LLM, and lending its memory to another process on this machine that needs it for a while (memory::Lender:
// POST /v1/memory/acquire). A loan waits until no LLM lease is out (LLM leases asked for meanwhile wait for it), then
// parks the conversations (gen::SessionPool::suspend) and unloads the LLM. The next LLM lease asks for the memory back,
// waits until the borrower has freed it, loads the LLM again and gives the pool its conversations back.
class Models {
public:
    struct Llm {
        std::unique_ptr<gen::Model> model;
        core::EngineBase* base = nullptr; // the same engine, for its expert cache
    };
    using LoadLlm = std::function<Llm()>;

    // Loads the LLM.
    explicit Models(LoadLlm load_llm);
    // An LLM the caller owns, never unloaded (its memory is not lent).
    explicit Models(gen::Model& fixed);
    ~Models();
    Models(const Models&) = delete;
    Models& operator=(const Models&) = delete;

    // The LLM as first loaded (a reload is the same model with the same options): for the pool and fixed figures.
    gen::Model& first_llm() { return *llm_model_; }
    // The pool whose conversations live across loans, and the identity they are kept under.
    void attach(gen::SessionPool& pool, std::string identity);
    // Whether the LLM's memory can be lent (it can be loaded again).
    bool can_lend() const { return bool(load_llm_); }
    memory::Lender& lender() { return *lender_; }

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

    // f(the LLM's engine) while the LLM is loaded; false (f not called) otherwise. Any thread.
    bool with_llm(const std::function<void(core::EngineBase&)>& f);
    // "llm", "lent" (to the process that borrowed the memory) or "none" (between an unload and a load).
    const char* resident();

private:
    void release_llm();
    // memory::Lender's hooks.
    void lend(const std::string& borrower);
    void returned();

    LoadLlm load_llm_;
    Llm llm_;
    gen::Model* llm_model_ = nullptr; // llm_.model's, or the fixed one; null while unloaded
    gen::SessionPool* pool_ = nullptr;
    std::string identity_;
    std::mutex mutex_; // the counts and what is loaded; held through a load
    std::condition_variable cv_;
    uint32_t llm_out_ = 0, lend_waiting_ = 0;
    bool lent_ = false;
    std::mutex llm_mutex_; // llm_.base for with_llm(), apart from a load's long hold
    std::atomic<const char*> resident_{"llm"};
    std::unique_ptr<memory::Lender> lender_;
};

} // namespace mako::server
