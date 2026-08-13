// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "prepare_whisper_model.hpp"

#include <optional>
#include <regex>

#include "../llm_compiled_model_utils.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/validate.hpp"

namespace opp = ov::pass::pattern;

namespace {

// diagnostics warnings on OPENVINO_MATCHER_PASS_RTTI() definition: visibility hidden
#ifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wattributes"
#endif

class AttentionMaskInputPast : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::LLMCompiledModel::AttentionMaskInputPast");

    AttentionMaskInputPast(std::shared_ptr<ov::Model> model) {
        auto range = opp::wrap_type<ov::op::v4::Range>();
        auto convert1 = opp::wrap_type<ov::op::v0::Convert>({range});
        auto greater = opp::wrap_type<ov::op::v1::Greater>({convert1, opp::any_input()});
        auto convert2 = opp::wrap_type<ov::op::v0::Convert>({greater});

        register_matcher(std::make_shared<opp::Matcher>(convert2, this->get_type_info().name),
                         [model](opp::Matcher& m) {
                             auto node = m.get_match_root();
                             auto attention_mask =
                                 std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
                             attention_mask->get_output_tensor(0).set_names({"attention_mask"});
                             model->add_parameters({attention_mask});

                             auto cvt =
                                 std::make_shared<ov::op::v0::Convert>(attention_mask->output(0), ov::element::f32);
                             ov::replace_node(node, cvt);
                             return false;
                         });
    }
};

class AttentionMaskInputPast_2 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::LLMCompiledModel::AttentionMaskInputPast_2");

    AttentionMaskInputPast_2(std::shared_ptr<ov::Model> model) {
        auto range = opp::wrap_type<ov::op::v4::Range>();
        auto unsqueeze1 = opp::wrap_type<ov::op::v0::Unsqueeze>({range, opp::any_input()});
        auto unsqueeze2 = opp::wrap_type<ov::op::v0::Unsqueeze>({unsqueeze1, opp::any_input()});
        auto unsqueeze3 = opp::wrap_type<ov::op::v0::Unsqueeze>({unsqueeze2, opp::any_input()});
        auto opt_convert = opp::optional<ov::op::v0::Convert>({unsqueeze3->output(0)});
        auto lessequal = opp::wrap_type<ov::op::v1::LessEqual>({opt_convert, opp::any_input()});

        register_matcher(
            std::make_shared<opp::Matcher>(lessequal, this->get_type_info().name),
            [model](opp::Matcher& m) {
                auto node = m.get_match_root();
                auto attention_mask =
                    std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});
                attention_mask->get_output_tensor(0).set_names({"attention_mask"});
                model->add_parameters({attention_mask});

                auto cst_0 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, 0);
                auto cst_1 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, 1);
                auto cst_2 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, 2);

                auto attn_mask_shape =
                    std::make_shared<ov::op::v3::ShapeOf>(attention_mask, ov::element::i64)->output(0);
                auto gather = std::make_shared<ov::op::v8::Gather>(attn_mask_shape, cst_1, cst_0)->output(0);
                auto attn_mask_size_minus_one = std::make_shared<ov::op::v1::Subtract>(gather, cst_1)->output(0);
                auto slice = std::make_shared<ov::op::v8::Slice>(attention_mask->output(0),
                                                                 cst_0,
                                                                 attn_mask_size_minus_one,
                                                                 cst_1,
                                                                 cst_1);

                auto unsqueeze_1 = std::make_shared<ov::op::v0::Unsqueeze>(slice->output(0), cst_1->output(0));
                auto unsqueeze_2 = std::make_shared<ov::op::v0::Unsqueeze>(unsqueeze_1->output(0), cst_2->output(0));

                auto equal = std::make_shared<ov::op::v1::Equal>(unsqueeze_2->output(0), cst_0->output(0));

                ov::replace_node(node, equal);
                return false;
            });
    }
};

// If GenAI has already decomposed cross-attention SDPA (for word-level timestamps),
// there's no SDPA node left to find for it. The decomposition always tags the "QK
// scaled scores" node with this well-known tensor name, so it doubles as a reliable
// marker for where the block lives.
std::vector<std::shared_ptr<ov::Node>> find_decomposed_cross_attn_score_nodes(
    const std::shared_ptr<ov::Model>& model) {
    std::vector<std::shared_ptr<ov::Node>> found;
    for (const auto& op : model->get_ordered_ops()) {
        bool matched = false;
        for (const auto& output : op->outputs()) {
            for (const auto& name : output.get_names()) {
                if (name.find("cross_attention_qk_scaled_scores") != std::string::npos) {
                    matched = true;
                    break;
                }
            }
            if (matched) {
                break;
            }
        }
        if (matched) {
            found.push_back(op);
        }
    }
    return found;
}

class AttentionMaskInput : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::LLMCompiledModel::AttentionMaskInput");

    AttentionMaskInput(std::shared_ptr<ov::Model> model,
                       const uint32_t& max_prompt_len,
                       const uint32_t& lhs_seq_size,
                       bool transform_cross_attn) {
        std::vector<std::shared_ptr<ov::Node>> self_attn_nodes;
        std::vector<std::shared_ptr<ov::Node>> cross_attn_nodes;
        const auto kAttnMaskPort = 3;
        for (auto node : model->get_ops()) {
            if (ov::is_type<ov::op::v13::ScaledDotProductAttention>(node)) {
                if (node->inputs().size() > kAttnMaskPort &&
                    (ov::is_type<ov::op::v8::Slice>(node->input(kAttnMaskPort).get_source_output().get_node()) ||
                     ov::is_type<ov::op::v1::Select>(node->input(kAttnMaskPort).get_source_output().get_node()))) {
                    self_attn_nodes.push_back(node);
                } else {
                    cross_attn_nodes.push_back(node);
                }
            }
        }
        auto decomposed_cross_attn_nodes =
            transform_cross_attn ? find_decomposed_cross_attn_score_nodes(model)
                                 : std::vector<std::shared_ptr<ov::Node>>{};

        // Self-attention
        OPENVINO_ASSERT(!self_attn_nodes.empty());

        auto attention_mask = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
        attention_mask->get_output_tensor(0).set_names({"attention_mask"});
        model->add_parameters({attention_mask});

        auto cst_ninf = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                               ov::Shape{1},
                                                               std::vector<float>{-std::numeric_limits<float>::max()});
        auto cst_1 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{1});
        auto cst_0 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{0});

        auto slice = self_attn_nodes[0]->input(kAttnMaskPort).get_source_output().get_node_shared_ptr();
        std::shared_ptr<ov::Node> slice_f32;
        if (slice->get_element_type() == ov::element::boolean) {
            slice_f32 = std::make_shared<ov::op::v1::Select>(slice->output(0), cst_0->output(0), cst_ninf->output(0));
        } else {
            slice_f32 = slice;
        }
        auto cvt = std::make_shared<ov::op::v0::Convert>(attention_mask->output(0), ov::element::f32);
        auto add = std::make_shared<ov::op::v1::Add>(slice_f32->output(0), cvt->output(0));

        auto trps = std::make_shared<ov::op::v1::Transpose>(
            cvt->output(0),
            ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, std::vector<int>{1, 0}));
        auto mtpl = std::make_shared<ov::op::v1::Multiply>(trps->output(0), add->output(0));

        auto equal = std::make_shared<ov::op::v1::Equal>(mtpl->output(0), cst_1->output(0));
        auto select = std::make_shared<ov::op::v1::Select>(equal->output(0), cst_0->output(0), cst_ninf->output(0));

        for (auto self_attn : self_attn_nodes) {
            self_attn->input(3).replace_source_output(select->output(0));
        }

        if (transform_cross_attn) {
            // Cross attn
            OPENVINO_ASSERT(!cross_attn_nodes.empty() || !decomposed_cross_attn_nodes.empty());
            // FIXME: Should be taken from topology - don't hardcode!!!
            auto shape_cst =
                std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                       ov::Shape{2},
                                                       std::vector<float>{static_cast<float>(max_prompt_len), 1});
            auto target_shape = std::make_shared<ov::op::v0::Constant>(
                ov::element::i64,
                ov::Shape{2},
                std::vector<float>{static_cast<float>(max_prompt_len), static_cast<float>(lhs_seq_size)});
            // FIXME: Must be transpose if batch present
            auto reshape = std::make_shared<ov::op::v1::Reshape>(cvt->output(0), shape_cst->output(0), false);
            auto equal = std::make_shared<ov::op::v1::Equal>(reshape->output(0), cst_1->output(0));
            auto select = std::make_shared<ov::op::v1::Select>(equal->output(0), cst_0->output(0), cst_ninf->output(0));
            auto broadcast = std::make_shared<ov::op::v3::Broadcast>(select->output(0), target_shape->output(0));
            auto unsq1 = std::make_shared<ov::op::v0::Unsqueeze>(broadcast->output(0), cst_0->output(0));
            auto unsq2 = std::make_shared<ov::op::v0::Unsqueeze>(unsq1->output(0), cst_1->output(0));
            for (auto cross_attn_node : cross_attn_nodes) {
                if (cross_attn_node->inputs().size() == 3) {
                    auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(
                        cross_attn_node->input(0).get_source_output(),
                        cross_attn_node->input(1).get_source_output(),
                        cross_attn_node->input(2).get_source_output(),
                        unsq2->output(0),
                        false);
                    ov::replace_node(cross_attn_node, sdpa);
                } else {
                    cross_attn_node->input(3).replace_source_output(unsq2->output(0));
                }
            }
            for (const auto& qk_score_node : decomposed_cross_attn_nodes) {
                if (ov::is_type<ov::op::v1::Add>(qk_score_node)) {
                    // The original SDPA already had a mask input (or was causal), so
                    // decomposition produced Add(scores, mask) - mirrors the "cross_attn_node
                    // already has 4/5 inputs" branch above.
                    qk_score_node->input(1).replace_source_output(unsq2->output(0));
                } else {
                    // Mirrors the "cross_attn_node has 3 inputs" branch above: the original
                    // SDPA had no mask at all, so decomposition produced no Add either - splice
                    // one in and move the "cross_attention_qk_scaled_scores*" tensor name(s)
                    // onto it, since that's the value word-level-timestamp extraction expects.
                    // Snapshot readers before wiring up new_add itself as a reader.
                    auto readers = qk_score_node->output(0).get_target_inputs();
                    auto new_add = std::make_shared<ov::op::v1::Add>(qk_score_node->output(0), unsq2->output(0));
                    new_add->set_friendly_name(qk_score_node->get_friendly_name() + "/with_mask");
                    auto names = qk_score_node->output(0).get_names();
                    qk_score_node->output(0).set_names({});
                    new_add->output(0).add_names(names);
                    for (const auto& reader : readers) {
                        reader.replace_source_output(new_add->output(0));
                    }
                }
            }
        }
    }
};

class CachePositionInput : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::LLMCompiledModel::CachePositionInput");

    CachePositionInput(std::shared_ptr<ov::Model> model) {
        // Here the past length is folded into Range's start:
        //
        //   Gather ---------------+
        //          \              +-> Range -> Unsqueeze -> Tile
        //           -> Add -------+
        //
        auto gather = opp::wrap_type<ov::op::v8::Gather>({opp::any_input(), opp::any_input(), opp::any_input()});
        auto add = opp::wrap_type<ov::op::v1::Add>({gather, opp::any_input()});
        auto range = opp::wrap_type<ov::op::v4::Range>({gather, add, opp::any_input()});
        auto unsqueeze = opp::wrap_type<ov::op::v0::Unsqueeze>({range, opp::any_input()});
        auto tile = opp::wrap_type<ov::op::v0::Tile>({unsqueeze, opp::any_input()});

        register_matcher(
            std::make_shared<opp::Matcher>(tile, this->get_type_info().name),
            [model, unsqueeze](opp::Matcher& m) {
                auto& node_to_output = m.get_pattern_value_map();
                auto unsqueeze_node = node_to_output.at(unsqueeze).get_node_shared_ptr();
                auto matched_unsqueeze = std::static_pointer_cast<ov::op::v0::Unsqueeze>(unsqueeze_node);

                auto cache_position = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
                cache_position->get_output_tensor(0).set_names({"cache_position"});
                cache_position->set_friendly_name("cache_position");
                model->add_parameters({cache_position});
                std::shared_ptr<ov::Node> cache_pos_unsqueeze_arg;
                if (matched_unsqueeze->input(0).get_element_type() == ov::element::f32) {
                    cache_pos_unsqueeze_arg = std::make_shared<ov::op::v0::Convert>(cache_position, ov::element::f32);
                } else {
                    cache_pos_unsqueeze_arg = cache_position;
                }

                matched_unsqueeze->input(0).replace_source_output(cache_pos_unsqueeze_arg->output(0));
                return false;
            });
    }
};

class CachePositionInput_2 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::LLMCompiledModel::CachePositionInput_2");

    CachePositionInput_2(std::shared_ptr<ov::Model> model) {
        // transformers>=5.4 aranges from zero and adds the past length on top of it instead of
        // folding it into Range's start as CachePositionInput above expects:
        //
        //   Range -> Add -> Unsqueeze -> Tile
        //
        auto range = opp::wrap_type<ov::op::v4::Range>();
        auto add = opp::wrap_type<ov::op::v1::Add>({range, opp::any_input()});
        auto unsqueeze = opp::wrap_type<ov::op::v0::Unsqueeze>({add, opp::any_input()});
        auto tile = opp::wrap_type<ov::op::v0::Tile>({unsqueeze, opp::any_input()});

        register_matcher(
            std::make_shared<opp::Matcher>(tile, this->get_type_info().name),
            [model, unsqueeze](opp::Matcher& m) {
                // Both patterns are rooted at Tile and neither claims the node it matched, so
                // GraphRewrite offers every Tile to both. Exactly one matches per model, but a
                // second match would add a second parameter named cache_position, of which only
                // the first would ever be found again.
                OPENVINO_ASSERT(!ov::npuw::util::has_input(model, "cache_position"),
                                "More than one subgraph matched a cache position pattern");

                auto& node_to_output = m.get_pattern_value_map();
                auto unsqueeze_node = node_to_output.at(unsqueeze).get_node_shared_ptr();
                auto matched_unsqueeze = std::static_pointer_cast<ov::op::v0::Unsqueeze>(unsqueeze_node);

                auto cache_position = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
                cache_position->get_output_tensor(0).set_names({"cache_position"});
                cache_position->set_friendly_name("cache_position");
                model->add_parameters({cache_position});
                std::shared_ptr<ov::Node> cache_pos_unsqueeze_arg;
                if (matched_unsqueeze->input(0).get_element_type() == ov::element::f32) {
                    cache_pos_unsqueeze_arg = std::make_shared<ov::op::v0::Convert>(cache_position, ov::element::f32);
                } else {
                    cache_pos_unsqueeze_arg = cache_position;
                }

                // cache_position is already absolute, so the matched Add is dropped with the
                // rest of the replaced subgraph.
                matched_unsqueeze->input(0).replace_source_output(cache_pos_unsqueeze_arg->output(0));
                return false;
            });
    }
};

// A cross-attention KV-cache state (the ReadValue producing the once-computed
// encoder key/value) is consumed differently depending on whether the model
// still has a fused cross-attention SDPA node, or whether GenAI has already
// decomposed cross-attention SDPA (for word-level timestamps):
//   - fused:      ReadValue -> [FakeConvert ->] SDPA (key at port 1, value at port 2)
//   - decomposed: key   feeds the Transpose that builds "kT" for QK^T (and,
//                 separately, a ShapeOf reading the same state to size that
//                 transpose - both are redirected, only the Transpose decides the role)
//                 value feeds the final (softmax @ value) MatMul directly
// Returns the role for a single direct reader of the ReadValue's output, or
// nullopt if that reader alone doesn't tell us (e.g. Assign, ShapeOf).
enum class EncoderKvRole { Key, Value };

std::optional<EncoderKvRole> classify_encoder_kv_reader(const ov::Input<ov::Node>& reader) {
    auto* node = reader.get_node();
    if (strstr(node->get_type_name(), "FakeConvert") != nullptr) {
        // fp8: ReadValue -> FakeConvert -> {SDPA | Transpose | MatMul}. FakeConvert has a
        // single consumer, so its role is exactly the role of that consumer.
        auto fc_readers = node->outputs()[0].get_target_inputs();
        OPENVINO_ASSERT(fc_readers.size() == 1);
        return classify_encoder_kv_reader(*fc_readers.begin());
    }
    if (strstr(node->get_type_name(), "ScaledDotProductAttention") != nullptr) {
        return reader.get_index() == 1 ? EncoderKvRole::Key : EncoderKvRole::Value;
    }
    if (strstr(node->get_type_name(), "Transpose") != nullptr) {
        return EncoderKvRole::Key;
    }
    if (strstr(node->get_type_name(), "MatMul") != nullptr) {
        return EncoderKvRole::Value;
    }
    return std::nullopt;
}

// Splits an encoder-attn KV-cache ReadValue's readers into its single Assign and the
// (one or more) readers that consume the state's value, and determines whether this
// state is "key" or "value" from among the latter.
struct EncoderKvState {
    std::shared_ptr<ov::op::v6::Assign> assign_node;
    std::vector<ov::Input<ov::Node>> value_readers;
    EncoderKvRole role;
};

EncoderKvState analyze_encoder_kv_read_value(const std::shared_ptr<ov::Node>& rv_node) {
    OPENVINO_ASSERT(rv_node->outputs().size() == 1);
    EncoderKvState state;
    std::optional<EncoderKvRole> role;
    for (const auto& reader : rv_node->output(0).get_target_inputs()) {
        if (strstr(reader.get_node()->get_type_name(), "Assign") != nullptr) {
            OPENVINO_ASSERT(!state.assign_node, "More than one Assign reads an encoder-attn KV-cache state");
            state.assign_node = ov::as_type_ptr<ov::op::v6::Assign>(reader.get_node()->shared_from_this());
            continue;
        }
        state.value_readers.push_back(reader);
        if (!role) {
            role = classify_encoder_kv_reader(reader);
        }
    }
    OPENVINO_ASSERT(state.assign_node, "encoder-attn KV-cache state has no Assign");
    OPENVINO_ASSERT(!state.value_readers.empty(), "encoder-attn KV-cache state is never read");
    OPENVINO_ASSERT(role, "Could not classify encoder-attn KV-cache state as key or value");
    state.role = *role;
    return state;
}

auto remove_encoder_attn_read_value(const std::shared_ptr<ov::Node>& rv_node, const EncoderKvState& state) {
    auto kv_out = rv_node->input_value(0);
    // Redirect every consumer of the state directly to its initial value - covers both
    // the single fused-SDPA reader and the several decomposed-key readers (Transpose +
    // ShapeOf) alike.
    for (const auto& reader : state.value_readers) {
        reader.replace_source_output(kv_out);
    }
    return std::make_pair(std::make_shared<ov::op::v0::Result>(kv_out), state.assign_node);
}

std::string transform_key_value_name(std::string input_string,
                                     std::string prefix,
                                     std::string enc_or_dec,
                                     std::string key_or_value) {
    std::regex pattern("[0-9]+");
    std::smatch match;
    std::regex_search(input_string, match, pattern);

    if (match.empty())
        OPENVINO_THROW("Input string does not match the expected pattern");

    auto number = std::string(match[0]);
    return prefix + "." + number + enc_or_dec + key_or_value;
}

void set_name(std::shared_ptr<ov::Node> result, const std::string& name) {
    result->set_friendly_name(name);
    result->get_output_tensor(0).set_names({name});
}

void expose_runtime_states_as_outputs(const std::shared_ptr<ov::Model>& model) {
    // Find all ReadValue nodes
    ov::NodeVector read_value_nodes;
    for (const auto& op : model->get_ops()) {
        if (strstr(op->get_type_name(), "ReadValue") != nullptr) {
            read_value_nodes.push_back(op);
        }
    }

    // Holds result layers for cross-attn KV-cache tensors
    ov::ResultVector results;
    ov::SinkVector assigns;

    // Go through all ReadValue nodes and remove them
    for (const auto& rv_node : read_value_nodes) {
        OPENVINO_ASSERT(rv_node->inputs().size() == 1);
        auto state = analyze_encoder_kv_read_value(rv_node);
        auto [result, assign] = remove_encoder_attn_read_value(rv_node, state);
        auto key_or_value = state.role == EncoderKvRole::Key ? "key" : "value";
        auto normalized_name = transform_key_value_name(rv_node->input_value(0).get_node()->get_friendly_name(),
                                                        "present",
                                                        ".encoder.",
                                                        key_or_value);
        set_name(result, normalized_name);
        results.push_back(result);
        assigns.push_back(assign);
    }

    // Add, remove, validate
    model->add_results(results);
    for (const auto& assign : assigns) {
        model->remove_sink(assign);
    }
    model->validate_nodes_and_infer_types();
}

void remove_cache_position(const std::shared_ptr<ov::Model>& model) {
    // Build subgraph that will replace cache_pos
    auto input_ids = model->input("input_ids").get_node();
    auto shape_of_node = std::make_shared<ov::op::v3::ShapeOf>(input_ids->outputs()[0]);

    std::vector<int> v_0{0};
    std::vector<int> v_1{1};

    auto indices = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, v_1);
    indices->set_friendly_name("indices");
    auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, v_0);
    axis->set_friendly_name("axis");

    auto gather_node = std::make_shared<ov::op::v8::Gather>(shape_of_node->outputs()[0], indices, axis);

    auto cst_node = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, v_0);
    auto step = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, v_1);
    step->set_friendly_name("step");
    auto range_node = std::make_shared<ov::op::v4::Range>(cst_node->outputs()[0],
                                                          gather_node->outputs()[0],
                                                          step->outputs()[0],
                                                          ov::element::i64);
    // Replace cache_position
    auto cache_pos =
        ov::as_type_ptr<ov::op::v0::Parameter>(model->input("cache_position").get_node()->shared_from_this());
    for (const auto& target_input : cache_pos->outputs()[0].get_target_inputs()) {
        target_input.replace_source_output(range_node->outputs()[0]);
    }

    model->remove_parameter(cache_pos);
    model->validate_nodes_and_infer_types();
}

void expose_runtime_states_as_inputs(const std::shared_ptr<ov::Model>& model) {
    // Store Assign nodes to perform remove_sink later on
    ov::SinkVector assigns;
    // To add new Params to the model
    ov::ParameterVector params;

    ov::NodeVector read_value_nodes;
    for (const auto& op : model->get_ops()) {
        if (strstr(op->get_type_name(), "ReadValue") != nullptr) {
            read_value_nodes.push_back(op);
        }
    }

    for (const auto& rv_node : read_value_nodes) {
        auto state = analyze_encoder_kv_read_value(rv_node);

        auto shape = rv_node->get_output_partial_shape(0);
        auto new_param = std::make_shared<ov::op::v0::Parameter>(rv_node->get_output_element_type(0), shape);
        auto key_or_value = state.role == EncoderKvRole::Key ? "key" : "value";
        // Layer index comes from the state's producer (the initial-value subgraph),
        // which is unaffected by whether cross-attention SDPA is fused or decomposed -
        // unlike the readers, whose names/types differ between the two shapes.
        auto normalized_name = transform_key_value_name(rv_node->input_value(0).get_node()->get_friendly_name(),
                                                        "past_key_values",
                                                        ".encoder.",
                                                        key_or_value);
        set_name(new_param, normalized_name);
        params.push_back(new_param);

        for (const auto& reader : state.value_readers) {
            reader.replace_source_output(new_param->output(0));
        }
        assigns.push_back(state.assign_node);
    }

    // Remove sinks and add new params
    model->add_parameters(params);
    for (const auto& assign : assigns) {
        model->remove_sink(assign);
    }
}

void normalize_input_key_value_names(const std::shared_ptr<ov::Model>& model) {
    ov::ResultVector new_results, old_results;
    for (const auto& in : model->inputs()) {
        if (in.get_any_name().find("decoder") == std::string::npos) {
            continue;
        }

        auto key_or_value = (in.get_any_name().find(".key") != std::string::npos) ? "key" : "value";
        auto normalized_name =
            transform_key_value_name(in.get_any_name(), "past_key_values", ".decoder.", key_or_value);
        set_name(in.get_node_shared_ptr(), normalized_name);
    }

    model->validate_nodes_and_infer_types();
}

void normalize_output_key_value_names(const std::shared_ptr<ov::Model>& model) {
    ov::ResultVector new_results, old_results;
    for (const auto& out : model->outputs()) {
        if (out.get_any_name().find("decoder") == std::string::npos) {
            continue;
        }

        auto key_or_value = (out.get_any_name().find(".key") != std::string::npos) ? "key" : "value";
        auto normalized_name = transform_key_value_name(out.get_any_name(), "present", ".decoder.", key_or_value);
        set_name(out.get_node_shared_ptr(), normalized_name);
    }

    model->validate_nodes_and_infer_types();
}

void add_attention_mask_input(const std::shared_ptr<ov::Model>& model,
                              const uint32_t& max_prompt_size = 0,
                              const uint32_t& lhs_seq_size = 0,
                              bool transform_cross_attn = false) {
    ov::pass::GraphRewrite rewr;
    if (transform_cross_attn) {
        rewr.add_matcher<AttentionMaskInput>(model, max_prompt_size, lhs_seq_size, transform_cross_attn);
    } else {
        rewr.add_matcher<AttentionMaskInputPast>(model);
        rewr.add_matcher<AttentionMaskInputPast_2>(model);  // transformers>=4.53
    }

    rewr.run_on_model(model);

    ov::pass::Validate().run_on_model(model);
}

void add_cache_position_input(const std::shared_ptr<ov::Model>& model) {
    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<CachePositionInput>(model);
    rewr.add_matcher<CachePositionInput_2>(model);  // transformers>=5.4
    rewr.run_on_model(model);

    ov::pass::Validate().run_on_model(model);
}

#ifdef __GNUC__
#    pragma GCC diagnostic pop
#endif

}  // namespace

bool ov::npuw::util::PrepareWhisperPrefillModel::run_on_model(const std::shared_ptr<ov::Model>& model) {
    // 2) Remove all non-runtime states from inputs (they empty on first iteration)
    // remove_input_kv_tensors(model); -> Done for LLM also
    // 3) Expose all states that requires initialization on the first run as outputs
    expose_runtime_states_as_outputs(model);
    // 4) Remove cache_position input if it exists
    if (has_input(model, "cache_position")) {
        remove_cache_position(model);
    }
    // 5) Normalize output names - should be done in stateful_to_stateless_transformation
    normalize_output_key_value_names(model);

    add_attention_mask_input(model, m_max_prompt_size, m_lhs_seq_size, true);

    model->validate_nodes_and_infer_types();

    return true;
}

bool ov::npuw::util::PrepareWhisperKVCacheModel::run_on_model(const std::shared_ptr<ov::Model>& model) {
    normalize_input_key_value_names(model);
    normalize_output_key_value_names(model);
    expose_runtime_states_as_inputs(model);

    if (!has_input(model, "cache_position")) {
        add_cache_position_input(model);
    }

    add_attention_mask_input(model);

    model->reshape({{"input_ids", ov::PartialShape({-1, 1})}});

    model->validate_nodes_and_infer_types();

    return true;
}

bool ov::npuw::util::has_decomposed_cross_attention_sdpa(const std::shared_ptr<ov::Model>& model) {
    return !find_decomposed_cross_attn_score_nodes(model).empty();
}
