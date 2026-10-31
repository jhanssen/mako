#include "server/models.hpp"

#include "log/log.hpp"

#include <chrono>
#include <format>
#include <stdexcept>

namespace mako::server {

namespace {

double secs_since(std::chrono::steady_clock::time_point t)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

} // namespace

Models::Models(LoadLlm load_llm) : load_llm_(std::move(load_llm)), llm_(load_llm_()), llm_model_(llm_.model.get())
{
    lender_ = std::make_unique<memory::Lender>(memory::Lender::Hooks{
        .lend = [this](const std::string& borrower) { lend(borrower); }, .returned = [this] { returned(); }});
}

Models::Models(gen::Model& fixed) : llm_model_(&fixed)
{
    lender_ = std::make_unique<memory::Lender>(memory::Lender::Hooks{
        .lend = [](const std::string&) { throw std::runtime_error("this server's LLM is not unloaded for others"); },
        .returned = [] {}});
}

Models::~Models() = default;

void Models::attach(gen::SessionPool& pool, std::string identity)
{
    pool_ = &pool;
    identity_ = std::move(identity);
}

Models::Lease<gen::Model> Models::llm()
{
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [&] { return lend_waiting_ == 0; });
    if (lent_) {
        log::info("asking for the memory back");
        const auto t0 = std::chrono::steady_clock::now();
        lender_->reclaim();
        cv_.wait(lock, [&] { return !lent_; });
        log::info("the memory is back ({:.1f} s)", secs_since(t0));
    }
    if (!llm_model_) {
        resident_ = "none";
        const auto t0 = std::chrono::steady_clock::now();
        log::info("loading the LLM again");
        Llm l = load_llm_();
        {
            std::lock_guard g(llm_mutex_);
            llm_ = std::move(l);
            llm_model_ = llm_.model.get();
        }
        if (pool_)
            pool_->resume(*llm_model_, identity_);
        resident_ = "llm";
        log::info("the LLM is back in {:.1f} s; its conversations kept", secs_since(t0));
    }
    ++llm_out_;
    return Lease<gen::Model>(*this, *llm_model_, &Models::release_llm);
}

void Models::lend(const std::string& borrower)
{
    if (!load_llm_)
        throw std::runtime_error("this server's LLM is not unloaded for others");
    std::unique_lock lock(mutex_);
    ++lend_waiting_;
    cv_.wait(lock, [&] { return llm_out_ == 0; });
    --lend_waiting_;
    if (llm_model_) {
        if (pool_)
            pool_->suspend();
        std::lock_guard g(llm_mutex_);
        llm_ = {};
        llm_model_ = nullptr;
    }
    lent_ = true;
    resident_ = "lent";
    log::info("the LLM unloaded for {}", borrower);
}

void Models::returned()
{
    {
        std::lock_guard lock(mutex_);
        lent_ = false;
        resident_ = llm_model_ ? "llm" : "none";
    }
    cv_.notify_all();
}

void Models::release_llm()
{
    {
        std::lock_guard lock(mutex_);
        --llm_out_;
    }
    cv_.notify_all();
}

bool Models::with_llm(const std::function<void(core::EngineBase&)>& f)
{
    std::lock_guard g(llm_mutex_);
    if (!llm_.base)
        return false;
    f(*llm_.base);
    return true;
}

const char* Models::resident()
{
    return resident_.load();
}

} // namespace mako::server
