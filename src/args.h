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
#include <memory>

namespace fasttext {

enum class model_name : int { cbow = 1, sg, sup };
enum class loss_name : int { hs = 1, ns, softmax, ova };
class Tree;
class TreeNode{
 private:
  std::string name_;
  TreeNode* parent_;
  std::vector<std::shared_ptr<TreeNode>> children_;
  int depth_;
  Tree* tree_;

 public:
  std::shared_ptr<TreeNode> getChild(const std::string name);
  std::shared_ptr<TreeNode> addChild(const std::string name);
  TreeNode* parent();
  std::string name();
  Tree* tree();
  int depth();
  TreeNode(const std::string name,Tree* tree);
};
class Args {
 protected:
  std::string lossToString(loss_name) const;
  std::string boolToString(bool) const;
  std::string modelToString(model_name) const;

 public:
  Args();
  std::string input;
  std::string output;
  double lr;
  int lrUpdateRate;
  int dim;
  int ws;
  int epoch;
  int minCount;
  int minCountLabel;
  int neg;
  int wordNgrams;
  loss_name loss;
  model_name model;
  int bucket;
  int minn;
  int maxn;
  int thread;
  double t;
  std::string label;
  int verbose;
  std::string pretrainedVectors;
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
};
} // namespace fasttext
