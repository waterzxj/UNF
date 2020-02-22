#include "tokenizer.h"
#include "string_util.h"

#include <fstream>

Tokenizer::Tokenizer(const std::string &vocab_path) {
    std::ifstream ifs(vocab_path);
    if (!ifs) {
        std::cerr << "Load vocab fail!!" << std::endl;
        return;
    }

    std::string line;
    uint32_t index = 0;
    while(getline(ifs, line)) {
        std::string tmp = string_util::trim(line);
        vocab_[tmp] = index;
        index += 1;
    }
}

void Tokenizer::tokenize(const std::string &text, std::vector<std::string>& segment) {
    string_util::split(text, " ", segment);
    return;
}

void Tokenizer::token2id(const std::vector<std::string> &segment, std::vector<float> &t2id) {
    for (uint32_t index=0; index < segment.size(); index++) {
        if (vocab_.find(segment[index]) != vocab_.end()) {
            t2id.push_back(vocab_[segment[index]]);
        }
        else {
            //hard code WARNING
            t2id.push_back(vocab_["<unk>"]);
        }
    }
    return;
}

uint32_t Tokenizer::get_pad_index() {
    return vocab_["<pad>"];
}
