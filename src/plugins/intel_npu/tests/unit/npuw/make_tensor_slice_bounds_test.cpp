// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>

#include "infer_request_utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"

namespace {

class SliceableTensor {
public:
    explicit SliceableTensor(const ov::Shape& shape) : m_tensor(ov::element::f32, shape) {}

    ov::SoPtr<ov::ITensor> impl() const {
        return ov::get_tensor_impl(m_tensor);
    }

private:
    ov::Tensor m_tensor;
};

// These tests pin the bounds checks in ov::npuw::util::make_tensor_slice(), the single choke point
// every KV-slice caller funnels through.
TEST(MakeTensorSliceBoundsTest, ValidSliceProducesExpectedShape) {
    SliceableTensor tensor(ov::Shape{1, 4, 8, 16});

    ov::SoPtr<ov::ITensor> slice;
    ASSERT_NO_THROW(slice = ov::npuw::util::make_tensor_slice(tensor.impl(), /*dim=*/2u, /*start=*/2u, /*end=*/5u));
    EXPECT_EQ(slice->get_shape(), (ov::Shape{1, 4, 3, 16}));
}

TEST(MakeTensorSliceBoundsTest, FullExtentSliceIsAccepted) {
    SliceableTensor tensor(ov::Shape{1, 4, 8, 16});

    ov::SoPtr<ov::ITensor> slice;
    ASSERT_NO_THROW(slice = ov::npuw::util::make_tensor_slice(tensor.impl(), /*dim=*/2u, /*start=*/0u, /*end=*/8u));
    EXPECT_EQ(slice->get_shape(), (ov::Shape{1, 4, 8, 16}));
}

TEST(MakeTensorSliceBoundsTest, AxisEqualToRankIsRejected) {
    SliceableTensor tensor(ov::Shape{1, 4, 8, 16});
    EXPECT_THROW(ov::npuw::util::make_tensor_slice(tensor.impl(), /*dim=*/4u, /*start=*/0u, /*end=*/1u),
                 ov::Exception);
}

TEST(MakeTensorSliceBoundsTest, AxisFarBeyondRankIsRejected) {
    SliceableTensor tensor(ov::Shape{1, 4, 8, 16});
    EXPECT_THROW(ov::npuw::util::make_tensor_slice(tensor.impl(), /*dim=*/0xFFFFFFFFu, /*start=*/0u, /*end=*/1u),
                 ov::Exception);
}

TEST(MakeTensorSliceBoundsTest, EndBeyondExtentIsRejected) {
    SliceableTensor tensor(ov::Shape{1, 4, 8, 16});
    EXPECT_THROW(ov::npuw::util::make_tensor_slice(tensor.impl(), /*dim=*/2u, /*start=*/0u, /*end=*/9u),
                 ov::Exception);
}

TEST(MakeTensorSliceBoundsTest, InvertedIntervalIsRejected) {
    SliceableTensor tensor(ov::Shape{1, 4, 8, 16});
    EXPECT_THROW(ov::npuw::util::make_tensor_slice(tensor.impl(), /*dim=*/2u, /*start=*/5u, /*end=*/2u),
                 ov::Exception);
}

// An i4 tensor of shape [1, H, S, 4] is 2 bytes per token; byte i holds the value i.
ov::Tensor make_i4_kv(size_t heads, size_t tokens) {
    ov::Tensor tensor(ov::element::i4, ov::Shape{1, heads, tokens, 4});
    auto* bytes = static_cast<uint8_t*>(tensor.data());
    for (size_t i = 0; i < tensor.get_byte_size(); ++i) {
        bytes[i] = static_cast<uint8_t>(i);
    }
    return tensor;
}

uint8_t byte_at(const ov::SoPtr<ov::ITensor>& tensor, size_t h, size_t s, size_t b) {
    const auto& strides = tensor->get_strides();
    return static_cast<const uint8_t*>(tensor->data())[h * strides[1] + s * strides[2] + b * strides[3]];
}

TEST(MakeTensorSliceSubByteTest, SliceAlongTokensViewsThePackedBytes) {
    auto tensor = make_i4_kv(2, 3);

    ov::SoPtr<ov::ITensor> slice;
    ASSERT_NO_THROW(slice = ov::npuw::util::make_tensor_slice(ov::get_tensor_impl(tensor), 2u, 1u, 3u));
    EXPECT_EQ(slice->get_element_type(), ov::element::u8);
    ASSERT_EQ(slice->get_shape(), (ov::Shape{1, 2, 2, 2}));
    for (size_t h = 0; h < 2; ++h) {
        for (size_t s = 0; s < 2; ++s) {
            for (size_t b = 0; b < 2; ++b) {
                EXPECT_EQ(byte_at(slice, h, s, b), h * 6 + (s + 1) * 2 + b);
            }
        }
    }
}

TEST(MakeTensorSliceSubByteTest, SliceAlongInnermostDimensionIsRejected) {
    auto tensor = make_i4_kv(2, 3);
    EXPECT_THROW(ov::npuw::util::make_tensor_slice(ov::get_tensor_impl(tensor), 3u, 0u, 2u), ov::Exception);
}

// The generate step pattern: copy the new token's i4 KV into a slice of the i4 cache.
TEST(MakeTensorSliceSubByteTest, CopyTokenIntoCacheSlice) {
    auto token = make_i4_kv(2, 1);
    ov::Tensor cache(ov::element::i4, ov::Shape{1, 2, 3, 4});
    std::fill_n(static_cast<uint8_t*>(cache.data()), cache.get_byte_size(), uint8_t{0xFF});

    auto cache_slice = ov::npuw::util::make_tensor_slice(ov::get_tensor_impl(cache), 2u, 1u, 2u);
    ASSERT_NO_THROW(ov::npuw::util::copy_tensor_by_dim(ov::get_tensor_impl(token), cache_slice, 2u, 2u));

    const auto* bytes = static_cast<const uint8_t*>(cache.data());
    for (size_t h = 0; h < 2; ++h) {
        for (size_t s = 0; s < 3; ++s) {
            for (size_t b = 0; b < 2; ++b) {
                const uint8_t expected = s == 1 ? static_cast<uint8_t>(h * 2 + b) : uint8_t{0xFF};
                EXPECT_EQ(bytes[h * 6 + s * 2 + b], expected) << "h=" << h << " s=" << s << " b=" << b;
            }
        }
    }
}

}  // namespace
