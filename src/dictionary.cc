/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "dictionary.h"

#include <assert.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

namespace fasttext {

const std::string Dictionary::EOS = "</s>";
const std::string Dictionary::BOW = "<";
const std::string Dictionary::EOW = ">";

Dictionary::Dictionary(std::shared_ptr<Args> args)
    : args_(args),
      word2int_(MAX_VOCAB_SIZE, -1),
      label2int_(MAX_LABEL_SIZE, -1),
      size_(0),
      labelSize_(0),
      nwords_(0),
      nlabels_(0),
      ntokens_(0),
      pruneidx_size_(-1) {}

Dictionary::Dictionary(std::shared_ptr<Args> args, std::istream& in)
    : args_(args),
      size_(0),
      labelSize_(0),
      nwords_(0),
      nlabels_(0),
      ntokens_(0),
      pruneidx_size_(-1) {
  load(in);
}

int32_t Dictionary::find(const std::string& w) const {
  return find(w, hash(w));
}

int32_t Dictionary::find(const std::string& w, uint32_t h) const {
  int32_t word2intsize = word2int_.size();
  int32_t id = h % word2intsize;
  while (word2int_[id] != -1 && words_[word2int_[id]].word != w) {
    id = (id + 1) % word2intsize;
  }
  return id;
}



void Dictionary::add(const std::string& w) {
  int32_t h = find(w);
  ntokens_++;
  if (word2int_[h] == -1) {
    entry e;
    e.word = w;
    e.count = 1;
    e.type = getType(w);
    words_.push_back(e);
    word2int_[h] = size_++;
  } else {
    words_[word2int_[h]].count++;
  }
}

int32_t Dictionary::nwords() const {
  return nwords_;
}

int32_t Dictionary::nlabels() const {
  return nlabels_;
}

int64_t Dictionary::ntokens() const {
  return ntokens_;
}

const std::vector<int32_t>& Dictionary::getSubwords(int32_t i) const {
  assert(i >= 0);
  assert(i < nwords_);
  return words_[i].subwords;
}

const std::vector<int32_t> Dictionary::getSubwords(
    const std::string& word) const {
  int32_t i = getId(word);
  if (i >= 0) {
    return getSubwords(i);
  }
  std::vector<int32_t> ngrams;
  if (word != EOS) {
    computeSubwords(BOW + word + EOW, ngrams);
  }
  return ngrams;
}

void Dictionary::getSubwords(
    const std::string& word,
    std::vector<int32_t>& ngrams,
    std::vector<std::string>& substrings) const {
  int32_t i = getId(word);
  ngrams.clear();
  substrings.clear();
  if (i >= 0) {
    ngrams.push_back(i);
    substrings.push_back(words_[i].word);
  }
  if (word != EOS) {
    computeSubwords(BOW + word + EOW, ngrams, &substrings);
  }
}

bool Dictionary::discard(int32_t id, real rand) const {
  assert(id >= 0);
  assert(id < nwords_);
  if (args_->model == model_name::sup) {
    return false;
  }
  return rand > pdiscard_[id];
}

int32_t Dictionary::getId(const std::string& w, uint32_t h) const {
  int32_t id = find(w, h);
  return word2int_[id];
}

int32_t Dictionary::getId(const std::string& w) const {
  int32_t h = find(w);
  return word2int_[h];
}

entry_type Dictionary::getType(int32_t id) const {
  assert(id >= 0);
  assert(id < size_);
  return words_[id].type;
}

entry_type Dictionary::getType(const std::string& w) const {
  return (w.find(args_->label) == 0) ? entry_type::label : entry_type::word;
}

std::string Dictionary::getWord(int32_t id) const {
  assert(id >= 0);
  assert(id < size_);
  return words_[id].word;
}

// The correct implementation of fnv should be:
// h = h ^ uint32_t(uint8_t(str[i]));
// Unfortunately, earlier version of fasttext used
// h = h ^ uint32_t(str[i]);
// which is undefined behavior (as char can be signed or unsigned).
// Since all fasttext models that were already released were trained
// using signed char, we fixed the hash function to make models
// compatible whatever compiler is used.
uint32_t Dictionary::hash(const std::string& str) const {
  uint32_t h = 2166136261;
  for (size_t i = 0; i < str.size(); i++) {
    h = h ^ uint32_t(int8_t(str[i]));
    h = h * 16777619;
  }
  return h;
}

void Dictionary::computeSubwords(
    const std::string& word,
    std::vector<int32_t>& ngrams,
    std::vector<std::string>* substrings) const {
  for (size_t i = 0; i < word.size(); i++) {
    std::string ngram;
    if ((word[i] & 0xC0) == 0x80) { // in UTF-8, all bytes that begin with the bit pattern 10 are subsequent bytes of a multi-byte sequence
      continue;
    }
    for (size_t j = i, n = 1; j < word.size() && n <= args_->maxn; n++) {
      ngram.push_back(word[j++]);
      // std::cout<<word[j];
      while (j < word.size() && (word[j] & 0xC0) == 0x80) {
        ngram.push_back(word[j++]);
        // std::cout<<word[j];
      }
      // std::cout<<",";
      if (n >= args_->minn && !(n == 1 && (i == 0 || j == word.size()))) {
        int32_t h = hash(ngram) % args_->bucket;
        // std::cout<<"push subword:"<<ngram<<",subwordHash:"<<h<<std::endl;
        pushHash(ngrams, h);
        if (substrings) {
          substrings->push_back(ngram);
        }
      }
    }
  }
}

void Dictionary::initNgrams() {
  for (size_t i = 0; i < size_; i++) {
    std::string word = BOW + words_[i].word + EOW;
    words_[i].subwords.clear();
    words_[i].subwords.push_back(i);
    if (words_[i].word != EOS) {
      computeSubwords(word, words_[i].subwords);
    }
  }
}

bool Dictionary::readWord(std::istream& in, std::string& word) const {
  int c;
  std::streambuf& sb = *in.rdbuf();
  word.clear();
  while ((c = sb.sbumpc()) != EOF) {
    if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' ||
        c == '\f' || c == '\0') {
      if (word.empty()) {
        if (c == '\n') {
          word += EOS;
          return true;
        }
        continue;
      } else {
        if (c == '\n')
          sb.sungetc();
        return true;
      }
    }
    word.push_back(c);
  }
  // trigger eofbit
  in.get();
  return !word.empty();
}

// 新增函数
int32_t Dictionary::findL(const std::string& w, uint32_t h) {
  int32_t label2int_size = label2int_.size();
  int32_t id = h % label2int_size;
  while (label2int_[id] != -1 && labels_[label2int_[id]].word != w) {
    id = (id + 1) % label2int_size;
  }
  return id;
}
void Dictionary::add(const std::string& w,const entry_type type){
  ntokens_++;
  // std::cout<<"ntokens:"<<ntokens_<<std::endl;
  if(type == entry_type::word){
    int32_t h = find(w);
    if (word2int_[h] == -1) {
      entry e;
      e.word = w;
      e.count = 1;
      e.type = entry_type::word;
      words_.push_back(e);
      word2int_[h] = size_++;
    } else {
      words_[word2int_[h]].count++;
    } 
  }
  else
  {
    int32_t h = findL(w,hash(w));
    // std::cout<<"labelhash:"<<h<<",word2intSize:"<<word2int_.size()<<std::endl;
    if (label2int_[h] == -1) {
      entry e;
      e.word = w;
      e.count = 1;
      e.type = entry_type::label;
      labels_.push_back(e);
      label2int_[h] = labelSize_++;
    } else {
      labels_[label2int_[h]].count++;
    } 
  }
 
}
void Dictionary::readFromArray(const std::vector<std::vector<std::string>> features,const std::vector<std::string> labels, std::vector<std::pair<std::string,float>> sample_weight){
  std::string word;
  int64_t minThreshold = 1;
  int x_size = features.size();
  int y_size = labels.size();
  if(x_size!= y_size)
    throw std::invalid_argument("the number of x must equals to the number of y!");
  for(int i=0; i< x_size; i++){
    int feature_size = features[i].size();
    for(int j=0;j<feature_size;j++){
      // std::cout<<"add the  word:"<<features[i][j]<<std::endl;
      add(features[i][j],entry_type::word);
      if (ntokens_ % 1000000 == 0 && args_->verbose > 1) {
        std::cerr << "\rRead " << ntokens_ / 1000000 << "M words" << std::flush;
      }
      if (size_ > 0.75 * MAX_VOCAB_SIZE) {
        minThreshold++;
        fitThreshold(minThreshold, minThreshold);
      }      
    }
    // std::cout<<"add the  label:"<<labels[i]<<std::endl;
    add(labels[i],entry_type::label);
    // std::cout<<"label was added:"<<labels[i]<<std::endl;

  }
  fitThreshold(args_->minCount, args_->minCountLabel);

  weightTable.clear();
  weightTable.shrink_to_fit();

  int index = 0;
  for (auto it = labels_.begin(); it != labels_.end(); ++it) {
    int32_t h = findL(it->word,hash(it->word));
    bool isFind =false;
    for(auto sit = sample_weight.begin(); sit!=sample_weight.end();sit++){
      if(sit->first == it->word){
        weightTable.push_back(sit->second);
        // std::cout<<sit->first<<","<<it->word<<",push"<<sit->second<<std::endl;
        isFind = true;
        break;
      }
    }
    if(!isFind){
      weightTable.push_back(args_->alpha);
      // std::cout<<"push1.0"<<std::endl;
    }
    // std::cout<<it->word<<":"<<weightTable[index++]<<std::endl;
  }  


  initTableDiscard();
  initNgrams();
  if (args_->verbose > 0) {
    std::cerr << "\rRead " << ntokens_ / 1000000 << "M words" << std::endl;
    std::cerr << "Number of words:  " << nwords_ << std::endl;
    std::cerr << "Number of labels: " << nlabels_ << std::endl;
  }
  if (size_ == 0) {
    throw std::invalid_argument(
        "Empty vocabulary. Try a smaller -minCount value.");
  }
}

void Dictionary::fitThreshold(int64_t t, int64_t tl){
  sort(words_.begin(), words_.end(), [](const entry& e1, const entry& e2) {
    return e1.count > e2.count;
  });
  sort(labels_.begin(), labels_.end(), [](const entry& e1, const entry& e2) {
    return e1.count > e2.count;
  });
  labels_.erase(
      remove_if(
          labels_.begin(),
          labels_.end(),
          [&](const entry& e) {
            return e.count < tl;
          }),
      labels_.end());    
  words_.erase(
      remove_if(
          words_.begin(),
          words_.end(),
          [&](const entry& e) {
            return e.count < t ;
          }),
      words_.end());
  words_.shrink_to_fit();
  labels_.shrink_to_fit();
  size_ = 0;
  labelSize_ = 0;
  nwords_ = words_.size();
  nlabels_ = labels_.size();
  std::fill(word2int_.begin(), word2int_.end(), -1);
  std::fill(label2int_.begin(), label2int_.end(), -1);
  for (auto it = words_.begin(); it != words_.end(); ++it) {
    int32_t h = find(it->word);
    word2int_[h] = size_++;
  }

  for (auto it = labels_.begin(); it != labels_.end(); ++it) {
    int32_t h = findL(it->word,hash(it->word));
    label2int_[h] = labelSize_++;
  }  
}
int32_t Dictionary::getFitLine(
    std::vector<std::string> x,
    std::string y,
    std::vector<int32_t>& words,
    std::vector<int32_t>& labels){
  std::vector<int32_t> word_hashes;
  int32_t ntokens = 0;

  words.clear();
  labels.clear();
  int32_t size = x.size();
  
  for(int32_t i=0; i<size; i++){
      ntokens++;
      uint32_t h = hash(x[i]);
      int32_t wid = getId(x[i],h);
      addSubwords(words, x[i], wid);
      word_hashes.push_back(h);

      // std::cout<<"word:"<<x[i]<<",wid:"<<wid<<std::endl;
      // for(int k=0; k<words.size();k++){
      //   std::cout<<"subword"<<k<<":"<<words[k]<<std::endl;
      // }
  }
  labels.push_back(label2int_[findL(y,hash(y))]);
  addWordNgrams(words, word_hashes, args_->wordNgrams);
  return ntokens;  
}

std::vector<char*> Dictionary::getLabels(){
  if(labelName_.size()< 1){
    for (auto it = labels_.begin(); it != labels_.end(); ++it) {
      int32_t len = it->word.length();
      char* p = new char[len+1];
      for(int32_t i=0;i<len;i++)
        p[i]=it->word[i];
      p[len] = '\0';
      labelName_.push_back(p);
    }
  }
  return labelName_;  
}
//


void Dictionary::readFromFile(std::istream& in) {
  std::string word;
  int64_t minThreshold = 1;
  while (readWord(in, word)) {
    add(word);
    if (ntokens_ % 1000000 == 0 && args_->verbose > 1) {
      std::cerr << "\rRead " << ntokens_ / 1000000 << "M words" << std::flush;
    }
    if (size_ > 0.75 * MAX_VOCAB_SIZE) {
      minThreshold++;
      threshold(minThreshold, minThreshold);
    }
  }
  threshold(args_->minCount, args_->minCountLabel);
  initTableDiscard();
  initNgrams();
  if (args_->verbose > 0) {
    std::cerr << "\rRead " << ntokens_ / 1000000 << "M words" << std::endl;
    std::cerr << "Number of words:  " << nwords_ << std::endl;
    std::cerr << "Number of labels: " << nlabels_ << std::endl;
  }
  if (size_ == 0) {
    throw std::invalid_argument(
        "Empty vocabulary. Try a smaller -minCount value.");
  }
}

void Dictionary::threshold(int64_t t, int64_t tl) {
  sort(words_.begin(), words_.end(), [](const entry& e1, const entry& e2) {
    if (e1.type != e2.type) {
      return e1.type < e2.type;
    }
    return e1.count > e2.count;
  });
  words_.erase(
      remove_if(
          words_.begin(),
          words_.end(),
          [&](const entry& e) {
            return (e.type == entry_type::word && e.count < t) ||
                (e.type == entry_type::label && e.count < tl);
          }),
      words_.end());
  words_.shrink_to_fit();
  size_ = 0;
  nwords_ = 0;
  nlabels_ = 0;
  std::fill(word2int_.begin(), word2int_.end(), -1);
  for (auto it = words_.begin(); it != words_.end(); ++it) {
    int32_t h = find(it->word);
    word2int_[h] = size_++;
    if (it->type == entry_type::word) {
      nwords_++;
    }
    if (it->type == entry_type::label) {
      nlabels_++;
    }
  }
}

void Dictionary::initTableDiscard() {
  pdiscard_.resize(size_);
  for (size_t i = 0; i < size_; i++) {
    real f = real(words_[i].count) / real(ntokens_);
    pdiscard_[i] = std::sqrt(args_->t / f) + args_->t / f;
  }
}

std::vector<int64_t> Dictionary::getCounts(entry_type type) const {
  std::vector<int64_t> counts;
  for (auto& w : words_) {
    if (w.type == type) {
      counts.push_back(w.count);
    }
  }
  return counts;
}

void Dictionary::addWordNgrams(
    std::vector<int32_t>& line,
    const std::vector<int32_t>& hashes,
    int32_t n) const {
  for (int32_t i = 0; i < hashes.size(); i++) {
    uint64_t h = hashes[i];
    for (int32_t j = i + 1; j < hashes.size() && j < i + n; j++) {
      h = h * 116049371 + hashes[j];
      pushHash(line, h % args_->bucket);
    }
  }
}

void Dictionary::addSubwords(
    std::vector<int32_t>& line,
    const std::string& token,
    int32_t wid) const {
  if (wid < 0) { // out of vocab
    if (token != EOS) {
      computeSubwords(BOW + token + EOW, line);
    }
  } else {
    if (args_->maxn <= 0) { // in vocab w/o subwords
      line.push_back(wid);
    } else { // in vocab w/ subwords
      const std::vector<int32_t>& ngrams = getSubwords(wid);
      line.insert(line.end(), ngrams.cbegin(), ngrams.cend());
    }
  }
}

void Dictionary::reset(std::istream& in) const {
  if (in.eof()) {
    in.clear();
    in.seekg(std::streampos(0));
  }
}

int32_t Dictionary::getLine(
    std::istream& in,
    std::vector<int32_t>& words,
    std::minstd_rand& rng) const {
  std::uniform_real_distribution<> uniform(0, 1);
  std::string token;
  int32_t ntokens = 0;

  reset(in);
  words.clear();
  while (readWord(in, token)) {
    int32_t h = find(token);
    int32_t wid = word2int_[h];
    if (wid < 0) {
      continue;
    }

    ntokens++;
    if (getType(wid) == entry_type::word && !discard(wid, uniform(rng))) {
      words.push_back(wid);
    }
    if (ntokens > MAX_LINE_SIZE || token == EOS) {
      break;
    }
  }
  return ntokens;
}

int32_t Dictionary::getLine(
    std::istream& in,
    std::vector<int32_t>& words,
    std::vector<int32_t>& labels) const {
  std::vector<int32_t> word_hashes;
  std::string token;
  int32_t ntokens = 0;

  reset(in);
  words.clear();
  labels.clear();
  while (readWord(in, token)) {
    uint32_t h = hash(token);
    int32_t wid = getId(token, h);
    entry_type type = wid < 0 ? getType(token) : getType(wid);

    ntokens++;
    if (type == entry_type::word) {
      addSubwords(words, token, wid);
      word_hashes.push_back(h);
    } else if (type == entry_type::label && wid >= 0) {
      labels.push_back(wid - nwords_);
    }
    if (token == EOS) {
      break;
    }
  }
  addWordNgrams(words, word_hashes, args_->wordNgrams);
  return ntokens;
}

void Dictionary::pushHash(std::vector<int32_t>& hashes, int32_t id) const {
  if (pruneidx_size_ == 0 || id < 0) {
    return;
  }
  if (pruneidx_size_ > 0) {
    if (pruneidx_.count(id)) {
      id = pruneidx_.at(id);
    } else {
      return;
    }
  }
  hashes.push_back(nwords_ + id);
}

std::string Dictionary::getLabel(int32_t lid) const {
  if (lid < 0 || lid >= nlabels_) {
    throw std::invalid_argument(
        "Label id is out of range [0, " + std::to_string(nlabels_) + "]");
  }
  return words_[lid + nwords_].word;
}

void Dictionary::save(std::ostream& out) const {
  out.write((char*)&size_, sizeof(int32_t));
  out.write((char*)&labelSize_, sizeof(int32_t));
  out.write((char*)&nwords_, sizeof(int32_t));
  out.write((char*)&nlabels_, sizeof(int32_t));
  out.write((char*)&ntokens_, sizeof(int64_t));
  out.write((char*)&pruneidx_size_, sizeof(int64_t));
  for (int32_t i = 0; i < size_; i++) {
    entry e = words_[i];
    out.write(e.word.data(), e.word.size() * sizeof(char));
    out.put(0);
    out.write((char*)&(e.count), sizeof(int64_t));
    out.write((char*)&(e.type), sizeof(entry_type));
  }
  for(int32_t i = 0; i< labelSize_; i++){
    entry e = labels_[i];
    out.write(e.word.data(), e.word.size() * sizeof(char));
    out.put(0);
    out.write((char*)&(e.count), sizeof(int64_t));
    out.write((char*)&(e.type), sizeof(entry_type));    
  }
  for (const auto pair : pruneidx_) {
    out.write((char*)&(pair.first), sizeof(int32_t));
    out.write((char*)&(pair.second), sizeof(int32_t));
  }
}

void Dictionary::load(std::istream& in) {
  words_.clear();
  in.read((char*)&size_, sizeof(int32_t));
  in.read((char*)&labelSize_, sizeof(int32_t));
  in.read((char*)&nwords_, sizeof(int32_t));
  in.read((char*)&nlabels_, sizeof(int32_t));
  in.read((char*)&ntokens_, sizeof(int64_t));
  in.read((char*)&pruneidx_size_, sizeof(int64_t));
  for (int32_t i = 0; i < size_; i++) {
    char c;
    entry e;
    while ((c = in.get()) != 0) {
      e.word.push_back(c);
    }
    in.read((char*)&e.count, sizeof(int64_t));
    in.read((char*)&e.type, sizeof(entry_type));
    words_.push_back(e);
  }
  for(int32_t i = 0; i< labelSize_; i++){
    char c;
    entry e;
    while ((c = in.get()) != 0) {
      e.word.push_back(c);
    }
    in.read((char*)&e.count, sizeof(int64_t));
    in.read((char*)&e.type, sizeof(entry_type));
    labels_.push_back(e);
  }
  pruneidx_.clear();
  for (int32_t i = 0; i < pruneidx_size_; i++) {
    int32_t first;
    int32_t second;
    in.read((char*)&first, sizeof(int32_t));
    in.read((char*)&second, sizeof(int32_t));
    pruneidx_[first] = second;
  }
  initTableDiscard();
  initNgrams();

  int32_t word2intsize = MAX_VOCAB_SIZE;
  word2int_.assign(word2intsize, -1);
  int32_t label2intsize = MAX_LABEL_SIZE;
  label2int_.assign(label2intsize,-1);
  for (int32_t i = 0; i < size_; i++) {
    word2int_[find(words_[i].word)] = i;
  }
  for (int32_t i = 0; i < labelSize_; i++) {
    label2int_[findL(labels_[i].word,hash(labels_[i].word))] = i;
  }
}

void Dictionary::init() {
  initTableDiscard();
  initNgrams();
}

void Dictionary::prune(std::vector<int32_t>& idx) {
  std::vector<int32_t> words, ngrams;
  for (auto it = idx.cbegin(); it != idx.cend(); ++it) {
    if (*it < nwords_) {
      words.push_back(*it);
    } else {
      ngrams.push_back(*it);
    }
  }
  std::sort(words.begin(), words.end());
  idx = words;

  if (ngrams.size() != 0) {
    int32_t j = 0;
    for (const auto ngram : ngrams) {
      pruneidx_[ngram - nwords_] = j;
      j++;
    }
    idx.insert(idx.end(), ngrams.begin(), ngrams.end());
  }
  pruneidx_size_ = pruneidx_.size();

  std::fill(word2int_.begin(), word2int_.end(), -1);

  int32_t j = 0;
  for (int32_t i = 0; i < words_.size(); i++) {
    if (getType(i) == entry_type::label ||
        (j < words.size() && words[j] == i)) {
      words_[j] = words_[i];
      word2int_[find(words_[j].word)] = j;
      j++;
    }
  }
  nwords_ = words.size();
  size_ = nwords_ + nlabels_;
  words_.erase(words_.begin() + size_, words_.end());
  initNgrams();
}

void Dictionary::dump(std::ostream& out) const {
  out << words_.size() << std::endl;
  for (auto it : words_) {
    std::string entryType = "word";
    if (it.type == entry_type::label) {
      entryType = "label";
    }
    out << it.word << " " << it.count << " " << entryType << std::endl;
  }
}

} // namespace fasttext
