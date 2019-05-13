/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "densematrix.h"

#include <exception>
#include <random>
#include <stdexcept>
#include <utility>

#include "utils.h"
#include "vector.h"

namespace fasttext {

DenseMatrix::DenseMatrix(binarization_name bn) : DenseMatrix(0, 0, bn) {}

DenseMatrix::DenseMatrix(int64_t m, int64_t n, binarization_name bn)
    : Matrix(m, n),
      data_(m * n),
      bn_(bn) {}

DenseMatrix::DenseMatrix(DenseMatrix&& other) noexcept
    : Matrix(other.m_, other.n_),
      data_(std::move(other.data_)),
      bn_(std::move(other.bn_)) {}

void DenseMatrix::zero() {
  std::fill(data_.begin(), data_.end(), 0.0);
}

void DenseMatrix::uniform(real a) {
  std::minstd_rand rng(1);
  std::uniform_real_distribution<> uniform(-a, a);
  for (int64_t i = 0; i < (m_ * n_); i++) {
    data_[i] = uniform(rng);
  }
}

void DenseMatrix::multiplyRow(const Vector& nums, int64_t ib, int64_t ie) {
  if (ie == -1) {
    ie = m_;
  }
  assert(ie <= nums.size());
  for (auto i = ib; i < ie; i++) {
    real n = nums[i - ib];
    if (n != 0) {
      for (auto j = 0; j < n_; j++) {
        at(i, j) *= n;
      }
    }
  }
}

void DenseMatrix::divideRow(const Vector& denoms, int64_t ib, int64_t ie) {
  if (ie == -1) {
    ie = m_;
  }
  assert(ie <= denoms.size());
  for (auto i = ib; i < ie; i++) {
    real n = denoms[i - ib];
    if (n != 0) {
      for (auto j = 0; j < n_; j++) {
        at(i, j) /= n;
      }
    }
  }
}

real DenseMatrix::l2NormRow(int64_t i) const {
  auto norm = 0.0;
  for (auto j = 0; j < n_; j++) {
    norm += at(i, j) * at(i, j);
  }
  if (std::isnan(norm)) {
    throw std::runtime_error("Encountered NaN.");
  }
  return std::sqrt(norm);
}

void DenseMatrix::l2NormRow(Vector& norms) const {
  assert(norms.size() == m_);
  for (auto i = 0; i < m_; i++) {
    norms[i] = l2NormRow(i);
  }
}

real DenseMatrix::dotRow(
    const Vector& vec,
    int64_t i,
    std::minstd_rand& rng,
    bool binarize) const {
  assert(i >= 0);
  assert(i < m_);
  assert(vec.size() == n_);
  real d = 0.0;
  if (!binarize) {
    for (int64_t j = 0; j < n_; j++) {
      d += at(i, j) * vec[j];
    }
  } else {
    switch (bn_) {
      case binarization_name::none:
        for (int64_t j = 0; j < n_; j++) {
          d += at(i, j) * vec[j];
        }
        break;
      case binarization_name::dbc:
        for (int64_t j = 0; j < n_; j++) {
          d += utils::binarize(at(i, j)) * vec[j];
        }
        break;
      case binarization_name::sbc:
        std::uniform_real_distribution<> uniform(0.0, 1.0);
        real p;
        for (int64_t j = 0; j < n_; j++) {
          if (vec[j] > 1e-5) {
            p = utils::sigmoid(at(i, j));
            d += utils::binarize(uniform(rng) < p) * vec[j];
          }
        }
        break;
    }
  }
  if (std::isnan(d)) {
    throw std::runtime_error("Encountered NaN.");
  }
  return d;
}

void DenseMatrix::addVectorToRow(const Vector& vec, int64_t i, real a) {
  assert(i >= 0);
  assert(i < m_);
  assert(vec.size() == n_);
  for (int64_t j = 0; j < n_; j++) {
    data_[i * n_ + j] += a * vec[j];
  }
}

void DenseMatrix::addRowToVector(
    Vector& x,
    int32_t i,
    std::minstd_rand& rng,
    bool binarize) const {
  assert(i >= 0);
  assert(i < this->size(0));
  assert(x.size() == this->size(1));
  if (!binarize) {
    for (int64_t j = 0; j < this->size(1); j++) {
      x[j] += at(i, j);
    }
  } else {
    switch (bn_) {
      case binarization_name::none:
        for (int64_t j = 0; j < this->size(1); j++) {
          x[j] += at(i, j);
        }
        break;
      case binarization_name::dbc:
        for (int64_t j = 0; j < this->size(1); j++) {
          x[j] += utils::binarize(at(i, j));
        }
        break;
      case binarization_name::sbc:
        std::uniform_real_distribution<> uniform(0.0, 1.0);
        real p;
        for (int64_t j = 0; j < this->size(1); j++) {
          p = utils::sigmoid(at(i, j));
          x[j] += utils::binarize(uniform(rng) < p);
        }
        break;
    }
  }
}

void DenseMatrix::addRowToVector(
    Vector& x,
    int32_t i,
    real a,
    std::minstd_rand& rng,
    bool binarize) const {
  assert(i >= 0);
  assert(i < this->size(0));
  assert(x.size() == this->size(1));
  if (!binarize) {
    for (int64_t j = 0; j < this->size(1); j++) {
      x[j] += a * at(i, j);
    }
  } else {
    switch (bn_) {
      case binarization_name::none:
        for (int64_t j = 0; j < this->size(1); j++) {
          x[j] += a * at(i, j);
        }
        break;
      case binarization_name::dbc:
        for (int64_t j = 0; j < this->size(1); j++) {
          x[j] += a * utils::binarize(at(i, j));
        }
        break;
      case binarization_name::sbc:
        std::uniform_real_distribution<> uniform(0.0, 1.0);
        real p;
        for (int64_t j = 0; j < this->size(1); j++) {
          p = utils::sigmoid(at(i, j));
          x[j] += a * utils::binarize(uniform(rng) < p);
        }
        break;
    }
  }
}

void DenseMatrix::save(std::ostream& out) const {
  out.write((char*)&bn_, sizeof(binarization_name));
  out.write((char*)&m_, sizeof(int64_t));
  out.write((char*)&n_, sizeof(int64_t));
  out.write((char*)data_.data(), m_ * n_ * sizeof(real));
}

void DenseMatrix::load(std::istream& in) {
  in.read((char*)&bn_, sizeof(binarization_name));
  in.read((char*)&m_, sizeof(int64_t));
  in.read((char*)&n_, sizeof(int64_t));
  data_ = std::vector<real>(m_ * n_);
  in.read((char*)data_.data(), m_ * n_ * sizeof(real));
}

void DenseMatrix::dump(std::ostream& out) const {
  out << m_ << " " << n_ << std::endl;
  for (int64_t i = 0; i < m_; i++) {
    for (int64_t j = 0; j < n_; j++) {
      if (j > 0) {
        out << " ";
      }
      out << at(i, j);
    }
    out << std::endl;
  }
};

} // namespace fasttext
