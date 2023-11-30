/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../../byte_literals.hpp"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/device/aligned_resource_adaptor.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/failure_callback_resource_adaptor.hpp>
#include <rmm/mr/device/limiting_resource_adaptor.hpp>
#include <rmm/mr/device/logging_resource_adaptor.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>
#include <rmm/mr/device/thread_safe_resource_adaptor.hpp>
#include <rmm/mr/device/tracking_resource_adaptor.hpp>
#include <rmm/mr/resource_ref.hpp>

#include <cuda/memory_resource>

#include <gtest/gtest.h>

#include <cstddef>
#include <type_traits>

using cuda_mr = rmm::mr::cuda_memory_resource;
using rmm::mr::aligned_resource_adaptor;
using rmm::mr::failure_callback_resource_adaptor;
using rmm::mr::limiting_resource_adaptor;
using rmm::mr::logging_resource_adaptor;
using rmm::mr::statistics_resource_adaptor;
using rmm::mr::thread_safe_resource_adaptor;
using rmm::mr::tracking_resource_adaptor;
using owning_wrapper = rmm::mr::owning_wrapper<aligned_resource_adaptor<cuda_mr>, cuda_mr>;

// explicit instantiations for test coverage purposes
template class rmm::mr::aligned_resource_adaptor<cuda_mr>;
template class rmm::mr::failure_callback_resource_adaptor<cuda_mr>;
template class rmm::mr::limiting_resource_adaptor<cuda_mr>;
template class rmm::mr::logging_resource_adaptor<cuda_mr>;
template class rmm::mr::statistics_resource_adaptor<cuda_mr>;
template class rmm::mr::thread_safe_resource_adaptor<cuda_mr>;
template class rmm::mr::tracking_resource_adaptor<cuda_mr>;

namespace rmm::test {

using adaptors = ::testing::Types<aligned_resource_adaptor<cuda_mr>>; /*,
                                   failure_callback_resource_adaptor<cuda_mr>,
                                   limiting_resource_adaptor<cuda_mr>,
                                   logging_resource_adaptor<cuda_mr>,
                                   owning_wrapper,
                                   statistics_resource_adaptor<cuda_mr>,
                                   thread_safe_resource_adaptor<cuda_mr>,
                                   tracking_resource_adaptor<cuda_mr>>;*/

static_assert(
  cuda::mr::resource_with<rmm::mr::aligned_resource_adaptor<cuda_mr>, cuda::mr::device_accessible>);
static_assert(cuda::mr::resource_with<rmm::mr::failure_callback_resource_adaptor<cuda_mr>,
                                      cuda::mr::device_accessible>);
static_assert(cuda::mr::resource_with<rmm::mr::limiting_resource_adaptor<cuda_mr>,
                                      cuda::mr::device_accessible>);
static_assert(
  cuda::mr::resource_with<rmm::mr::logging_resource_adaptor<cuda_mr>, cuda::mr::device_accessible>);
static_assert(
  cuda::mr::resource_with<rmm::mr::owning_wrapper<cuda_mr>, cuda::mr::device_accessible>);
static_assert(cuda::mr::resource_with<rmm::mr::statistics_resource_adaptor<cuda_mr>,
                                      cuda::mr::device_accessible>);
static_assert(cuda::mr::resource_with<rmm::mr::thread_safe_resource_adaptor<cuda_mr>,
                                      cuda::mr::device_accessible>);
static_assert(cuda::mr::resource_with<rmm::mr::tracking_resource_adaptor<cuda_mr>,
                                      cuda::mr::device_accessible>);

template <typename MemoryResourceType>
struct AdaptorTest : public ::testing::Test {
  using adaptor_type = MemoryResourceType;

  std::unique_ptr<cuda_mr> cuda;
  rmm::device_resource_ref cuda_ref{cuda.get()};

  std::shared_ptr<adaptor_type> mr;

  AdaptorTest() : mr{make_adaptor(cuda_ref)} {}

  auto make_adaptor(rmm::device_resource_ref upstream)
  {
    if constexpr (std::is_same_v<adaptor_type, failure_callback_resource_adaptor<cuda_mr>>) {
      return std::make_shared<adaptor_type>(
        upstream, [](std::size_t bytes, void* arg) { return false; }, nullptr);
    } else if constexpr (std::is_same_v<adaptor_type, limiting_resource_adaptor<cuda_mr>>) {
      return std::make_shared<adaptor_type>(upstream, 64_MiB);
    } else if constexpr (std::is_same_v<adaptor_type, logging_resource_adaptor<cuda_mr>>) {
      return std::make_shared<adaptor_type>(upstream, "rmm_adaptor_test_log.txt");
    } else if constexpr (std::is_same_v<adaptor_type, owning_wrapper>) {
      return mr::make_owning_wrapper<aligned_resource_adaptor>(std::make_shared<cuda_mr>());
    } else {
      return std::make_shared<adaptor_type>(upstream);
    }
  }
};

TYPED_TEST_CASE(AdaptorTest, adaptors);

TYPED_TEST(AdaptorTest, Equality)
{
  std::cout << "Before is_equal\n";
  EXPECT_TRUE(this->mr->is_equal(*this->mr));
  std::cout << "After is_equal\n";

  {
    auto other_mr = this->make_adaptor(this->cuda_ref);
    std::cout << "Before is_equal\n";
    EXPECT_TRUE(this->mr->is_equal(*other_mr));
    std::cout << "After is_equal\n";
  }

  {
    rmm::device_resource_ref device_mr = this->cuda_ref;
    auto other_mr = aligned_resource_adaptor<rmm::mr::device_memory_resource>{device_mr};
    std::cout << "Before is_equal\n";
    EXPECT_FALSE(this->mr->is_equal(other_mr));
    std::cout << "After is_equal\n";
  }
}

TYPED_TEST(AdaptorTest, GetUpstream)
{
  if constexpr (std::is_same_v<TypeParam, owning_wrapper>) {
    EXPECT_TRUE(this->mr->wrapped().get_upstream() == this->cuda_ref);
  } else {
    EXPECT_TRUE(this->mr->get_upstream() == this->cuda_ref);
  }
}

TYPED_TEST(AdaptorTest, SupportsStreams)
{
  EXPECT_EQ(this->mr->supports_streams(), legacy(this->cuda_ref)->supports_streams());
}

TYPED_TEST(AdaptorTest, MemInfo)
{
  EXPECT_EQ(this->mr->supports_get_mem_info(), legacy(this->cuda_ref)->supports_get_mem_info());

  auto [free, total] = this->mr->get_mem_info(rmm::cuda_stream_default);

  if (this->mr->supports_get_mem_info()) {
    EXPECT_NE(total, 0);
  } else {
    EXPECT_EQ(free, 0);
    EXPECT_EQ(total, 0);
  }
}

TYPED_TEST(AdaptorTest, AllocFree)
{
  void* ptr{nullptr};
  EXPECT_NO_THROW(ptr = this->mr->allocate(1024));
  EXPECT_NE(ptr, nullptr);
  EXPECT_NO_THROW(this->mr->deallocate(ptr, 1024));
}

}  // namespace rmm::test
