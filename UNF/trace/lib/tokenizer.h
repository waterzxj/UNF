#pragma once

#include <string>
#include <vector>
#include <unordered_map>


class Tokenizer {
public:
    Tokenizer(const std::string &vocab_path);
    void token2id(const std::vector<std::string> &segment, std::vector<float> &t2id);
    void tokenize(const std::string &text, std::vector<std::string>& segment);
    uint32_t get_pad_index();

private:
    std::unordered_map<std::string, uint32_t> vocab_;
};
