/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <istream>
#include <ostream>

#include <memory>
#include <random>
#include <vector>

#include "real.h"

#include "densematrix.h"
#include "matrix.h"
#include "vector.h"

#include "productquantizer.h"

namespace fasttext {

class QuantMatrix : public Matrix {
 protected:
  std::unique_ptr<ProductQuantizer> pq_;
  std::unique_ptr<ProductQuantizer> npq_;

  std::vector<uint8_t> codes_;
  std::vector<uint8_t> norm_codes_;

  bool qnorm_;
  int32_t codesize_;

 public:
  QuantMatrix();
  QuantMatrix(DenseMatrix&&, int32_t, bool);
  QuantMatrix(const QuantMatrix&) = delete;
  QuantMatrix(QuantMatrix&&) = delete;
  QuantMatrix& operator=(const QuantMatrix&) = delete;
  QuantMatrix& operator=(QuantMatrix&&) = delete;
  virtual ~QuantMatrix() noexcept override = default;

  void quantizeNorm(const Vector&);
  void quantize(DenseMatrix&& mat);

  real dotRow(const Vector&, int64_t, std::minstd_rand&, bool=true) const override;
  real dotRow(int64_t, int64_t) const override;
  void addVectorToRow(const Vector&, int64_t, real) override;
  void addRowToRow(int64_t, int64_t, real) override;
  void addRowToVector(Vector& x, int32_t i, std::minstd_rand&, bool=true, bool=false, real=1.0) const override;
  void addRowToVector(Vector& x, int32_t i, real a, std::minstd_rand&, bool=true, bool=false, real=1.0) const override;
  void save(std::ostream&) const override;
  void load(std::istream&) override;
  void dump(std::ostream&) const override;
};

} // namespace fasttext
