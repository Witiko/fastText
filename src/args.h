/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <istream>
#include <ostream>
#include <string>
#include <vector>

namespace fasttext {

enum class model_name : int { cbow = 1, sg, sup };
enum class loss_name : int { hs = 1, ns, softmax, ova };
enum class binarization_name : int { none = 1, sbc, dbc };

class Args {
 protected:
  std::string lossToString(loss_name) const;
  std::string boolToString(bool) const;
  std::string modelToString(model_name) const;
  std::string binarizationToString(binarization_name) const;

 public:
  Args();
  std::string input;
  std::string output;
  double l2reg;
  double lr;
  int lrUpdateRate;
  int dim;
  int ws;
  double epoch;
  double epochSkip;
  double epochTotal;
  int minCount;
  int minCountLabel;
  int neg;
  int wordNgrams;
  loss_name loss;
  model_name model;
  binarization_name binarization;
  bool binarizeHidden;
  int bucket;
  int minn;
  int maxn;
  int thread;
  double t;
  std::string label;
  int verbose;
  std::string pretrainedVectors;
  std::string pretrainedModel;
  bool saveOutput;

  bool qout;
  bool retrain;
  bool qnorm;
  size_t cutoff;
  size_t dsub;

  void parseArgs(const std::vector<std::string>& args);
  void printHelp();
  void printBasicHelp();
  void printDictionaryHelp();
  void printTrainingHelp();
  void printQuantizationHelp();
  void save(std::ostream&);
  void load(std::istream&);
  void dump(std::ostream&) const;

  static constexpr const double implicit = -1.0;
};
} // namespace fasttext
