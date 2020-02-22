#include "string_util.h"
#include <stdio.h>

void string_util::vector2str(const std::vector<std::string>& vec, std::string& desc, const std::string& seq) {
    if (vec.size() <= 0) return;

    auto iter = vec.begin();
    if (iter != vec.end()) {
        desc.append(*iter);
        ++iter;
    }

    while (iter != vec.end()) {
        desc.append(seq);
        desc.append(*iter);
        ++iter;
    }
}

void string_util::split(const std::string& text, const std::string& seq, std::vector<std::string>& desc) {
    if (text.size() == 0) return;

    size_t seq_s = seq.size();
    size_t text_s = text.size();

    size_t start = 0;

    std::string tmp;
    while (start < text_s) {
        size_t pos = text.find_first_of(seq, start);

        if (pos != std::string::npos) {
            tmp = text.substr(start, pos - start);
            if (!tmp.empty()) {
                desc.push_back(tmp);
            }
            start = pos + seq_s;
        } else {
            break;
        }
    }

    if (start < text_s) {
        tmp = text.substr(start);
        if (!tmp.empty()) {
            desc.push_back(tmp);
        }
    }
    return;
}

std::string string_util::trim(const std::string& src, const char seq) {
    if (src.empty()) return src;
    std::string tmp = src;

    size_t len = tmp.size(), pos = 0;
    
    for (size_t i = 0; i < len; i++) {
        if (seq == src[i]) pos += 1;
        else {
            break;
        }
    }

    if (pos > 0) {
        tmp.erase(0, pos);
    }

    pos = 0;
    len = tmp.size();
    int i = int(len - 1);
    for (; i >= 0; i--) {
        if (seq == src[i]) {
            pos += 1;
        } else {
            break;
        }
    }

    if (pos > 0) {
        tmp.erase(len - pos);
    }

    return tmp;
}


int string_util::parse_char(const char *str) {
    if (str == NULL) {
        return 0;
    }

    unsigned char p = (unsigned char)(*str);
    int n = 0;
    while (p & 0x80) {
        ++n;
        p = p << 1;
    }

    if (n == 0) {
        ++n;
    } else if (n > 4) {
        n = 1;
    }

    return n;
}

int string_util::punct_process(const std::string &raw_str, std::string &norm_str, const std::string &replacer) {
    if (raw_str.empty()) {
        return -1;
    }

    norm_str.clear();
    char *p = (char *)raw_str.c_str();
    size_t offset = 0;
    int len = 0;
    while (*p) {
        len = parse_char(p);
        if (len == 1) {
            if (ispunct(*p)) {
                norm_str += replacer;
            } else {
                norm_str.push_back(*p);
            }
        } else if (len > 1) {
            std::string c = raw_str.substr(offset, len);
            if (is_cn_punct(c)) {
                norm_str += replacer;
            } else {
                norm_str += c;
            }
        }
        p += len;
        offset += len;
    }
    
    norm_str = trim(norm_str);
    return 0;
}

bool string_util::is_cn_punct(const std::string &word) {
    if (word.empty()) {
        return false;
    }

    if (CN_PUNCS.find(word) != CN_PUNCS.end()) {
        return true;
    } else {
        return false;
    }
}
