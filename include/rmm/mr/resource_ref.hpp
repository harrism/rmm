/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#pragma once

#include <cuda/memory_resource>

namespace rmm {

// forward decl
namespace mr {
class device_memory_resource;
}  // namespace mr

/// @brief property to provide access to legacy device_memory_resource from device_resource_ref
struct legacy_device_mr {
  using value_type = rmm::mr::device_memory_resource*;
};

/**
 * @brief alias for async_resource_ref with device_accessible and rmm::legacy_device_mr properties.
 *
 * This is a convenience for passing RMM memory resources.
 */
using device_resource_ref =
  cuda::mr::async_resource_ref<cuda::mr::device_accessible, legacy_device_mr>;

/**
 * @brief Helper to get the legacy device_memory_resource from a device_resource_ref.
 */
[[nodiscard]] inline rmm::mr::device_memory_resource* legacy(device_resource_ref const& ref)
{
  return get_property(ref, legacy_device_mr{});
}

}  // namespace rmm
