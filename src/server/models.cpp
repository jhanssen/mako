#include "server/models.hpp"

#include "core/log.hpp"

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

Models::Models(LoadLlm load_llm, LoadImage load_image)
    : load_llm_(std::move(load_llm)), load_image_(std::move(load_image)), llm_(load_llm_()), llm_model_(llm_.model.get())
{
}

Models::Models(gen::Model& fixed) : llm_model_(&fixed) {}

Models::~Models() = default;

void Models::attach(gen::SessionPool& pool, std::string identity)
{
    pool_ = &pool;
    identity_ = std::move(identity);
}

Models::Lease<gen::Model> Models::llm()
{
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [&] { return !image_out_ && image_waiting_ == 0; });
    if (!llm_model_) {
        resident_ = "none";
        image_.reset();
        {
            std::lock_guard g(name_mutex_);
            loaded_name_.clear();
        }
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

Models::Lease<image::Engine> Models::image(const ImageModel& model, const image::Loras& loras)
{
    if (!load_image_)
        throw std::runtime_error("this server has no image model (start it with --image-model)");
    std::unique_lock lock(mutex_);
    ++image_waiting_;
    cv_.wait(lock, [&] { return llm_out_ == 0 && !image_out_; });
    --image_waiting_;
    std::string key = model.key();
    for (const auto& [path, scale] : loras)
        key += std::format("|{}:{}", path, scale);
    if (image_ && image_key_ != key) {
        resident_ = "none";
        image_.reset();
        {
            std::lock_guard g(name_mutex_);
            loaded_name_.clear();
        }
        log::info("{} asked for: loading it", model.name == image_name_ ? "other LoRAs" : "image model " + model.name);
    }
    if (!image_) {
        const auto t0 = std::chrono::steady_clock::now();
        const bool swap = llm_model_ != nullptr;
        if (swap) {
            if (pool_)
                pool_->suspend();
            std::lock_guard g(llm_mutex_);
            llm_ = {};
            llm_model_ = nullptr;
        }
        resident_ = "none";
        try {
            image_ = load_image_(model, loras);
            image_key_ = key;
            image_name_ = model.name;
            std::lock_guard g(name_mutex_);
            loaded_name_ = model.name;
        } catch (...) {
            cv_.notify_all();
            throw;
        }
        resident_ = "image";
        log::info("image model {} loaded in {:.1f} s{}", model.name, secs_since(t0), swap ? " (the LLM unloaded)" : "");
    }
    image_out_ = true;
    return Lease<image::Engine>(*this, *image_, &Models::release_image);
}

void Models::release_llm()
{
    {
        std::lock_guard lock(mutex_);
        --llm_out_;
    }
    cv_.notify_all();
}

void Models::release_image()
{
    {
        std::lock_guard lock(mutex_);
        image_out_ = false;
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

std::string Models::image_name()
{
    std::lock_guard g(name_mutex_);
    return loaded_name_;
}

const char* Models::resident()
{
    return resident_.load();
}

} // namespace mako::server
