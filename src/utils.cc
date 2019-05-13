/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "utils.h"

#include <ios>

namespace fasttext {

namespace utils {

std::vector<real> t_sigmoid;

real sigmoid(real x) {
  if (x < -MAX_SIGMOID) {
    return 0.0;
  } else if (x > MAX_SIGMOID) {
    return 1.0;
  } else {
    int64_t i =
        int64_t((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
    return t_sigmoid[i];
  }
}

int64_t size(std::ifstream& ifs) {
  ifs.seekg(std::streamoff(0), std::ios::end);
  return ifs.tellg();
}

void seek(std::ifstream& ifs, int64_t pos) {
  ifs.clear();
  ifs.seekg(std::streampos(pos));
}

real binarize(bool x) {
  real sign = x ? 1 : -1;
  return sign / 3;
}

real binarize(real x) {
  real sign = x < 0 ? -1 : 1;
  return sign / 3;
}

} // namespace utils

} // namespace fasttext
