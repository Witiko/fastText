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
#include <random>
#include <vector>

#include <assert.h>
#include "args.h"
#include "matrix.h"
#include "real.h"

namespace fasttext {

class Vector;

class DenseMatrix : public Matrix {
 protected:
  std::vector<real> data_;
  binarization_name bn_;

 public:
  DenseMatrix(binarization_name);
  explicit DenseMatrix(int64_t, int64_t, binarization_name);
  DenseMatrix(const DenseMatrix&) = default;
  DenseMatrix(DenseMatrix&&) noexcept;
  DenseMatrix& operator=(const DenseMatrix&) = delete;
  DenseMatrix& operator=(DenseMatrix&&) = delete;
  virtual ~DenseMatrix() noexcept override = default;

  inline real* data() {
    return data_.data();
  }
  inline const real* data() const {
    return data_.data();
  }

  inline const real& at(int64_t i, int64_t j) const {
    assert(i * n_ + j < data_.size());
    return data_[i * n_ + j];
  };
  inline real& at(int64_t i, int64_t j) {
    return data_[i * n_ + j];
  };

  inline int64_t rows() const {
    return m_;
  }
  inline int64_t cols() const {
    return n_;
  }
  void zero();
  void uniform(real);

  void multiplyRow(const Vector& nums, int64_t ib = 0, int64_t ie = -1);
  void divideRow(const Vector& denoms, int64_t ib = 0, int64_t ie = -1);

  real l2NormRow(int64_t i) const;
  void l2NormRow(Vector& norms) const;

  real dotRow(const Vector&, int64_t, std::minstd_rand&, bool=true) const override;
  real dotRow(int64_t, int64_t) const override;
  void addVectorToRow(const Vector&, int64_t, real) override;
  void addRowToRow(int64_t, int64_t, real) override;
  void addRowToVector(Vector& x, int32_t i, std::minstd_rand&, bool=true) const override;
  void addRowToVector(Vector& x, int32_t i, real a, std::minstd_rand&, bool=true) const override;
  void save(std::ostream&) const override;
  void load(std::istream&) override;
  void dump(std::ostream&) const override;
};
} // namespace fasttext
