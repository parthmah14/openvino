// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "openvino/core/model.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::npuw {

ov::element::Type optimize_kv_cache_storage(const std::shared_ptr<ov::Model>& model);

// Element type the value cache is stored in for the given KV_CACHE_PRECISION and
// NPUW_LLM_KV_CACHE_COMPRESSION config.
ov::element::Type kv_cache_value_storage_type(const ov::element::Type kv_cache_precision,
                                              const std::string& kv_cache_compression_config);

class ConvertKVCacheToPrecision : public ov::pass::ModelPass {
    ov::element::Type m_lp_type;
    bool m_v_tensors_transposed = false;

    std::string m_kv_cache_compression_config;

public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::ConvertKVCacheToPrecision");
    explicit ConvertKVCacheToPrecision(const ov::element::Type lptype,
                                       bool v_tensors_transposed = false,
                                       const std::string& kv_cache_compression_config = "");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::npuw
