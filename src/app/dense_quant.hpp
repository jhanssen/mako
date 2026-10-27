#pragma once

#include "gguf/gguf.hpp"

#include <stdexcept>
#include <string>

namespace mako::app {

// --dense-quant's value: the format the dense Q8_0 weights are re-quantized to at load.
inline gguf::Type parse_dense_quant(const std::string& v)
{
    if (v == "q4_k")
        return gguf::Type::Q4_K;
    if (v == "q5_k")
        return gguf::Type::Q5_K;
    if (v == "q6_k")
        return gguf::Type::Q6_K;
    throw std::invalid_argument("takes q4_k, q5_k or q6_k");
}

} // namespace mako::app
