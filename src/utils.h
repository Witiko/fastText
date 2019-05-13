/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "real.h"

#include <algorithm>
#include <fstream>
#include <vector>

#if defined(__clang__) || defined(__GNUC__)
#define FASTTEXT_DEPRECATED(msg) __attribute__((__deprecated__(msg)))
#elif defined(_MSC_VER)
#define FASTTEXT_DEPRECATED(msg) __declspec(deprecated(msg))
#else
#define FASTTEXT_DEPRECATED(msg)
#endif

namespace fasttext {

using Predictions = std::vector<std::pair<real, int32_t>>;

namespace utils {

constexpr int64_t SIGMOID_TABLE_SIZE = 512;
constexpr int64_t MAX_SIGMOID = 8;

extern std::vector<real> t_sigmoid;

real sigmoid(real x);

int64_t size(std::ifstream&);

void seek(std::ifstream&, int64_t);

template <typename T>
bool contains(const std::vector<T>& container, const T& value) {
  return std::find(container.begin(), container.end(), value) !=
      container.end();
}

real binarize(real x);
real binarize(bool x);

} // namespace utils

} // namespace fasttext
