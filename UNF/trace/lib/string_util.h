#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <map>
#include <algorithm>
#include <set>
#include <unordered_set>


namespace string_util {

const static std::unordered_set<std::string> CN_PUNCS = {"，", "。", "...",
                   "；", "？", "|", "！", "_", "：", "“", "”", "《", "》",
                    "】", "【", "（", "）", "^"};
/*
 * @desc: 把vector的内容转换成string
 */
void vector2str(const std::vector<std::string>& vec, std::string& desc, const std::string& seq);

/*
 * @desc: 把text按照seq分隔符切分成vec，存进desc
 */
void split(const std::string& text, const std::string& seq, std::vector<std::string>& desc);

/*
 * @desc: string trim todo:空间复杂度o(n)太高
 */
std::string trim(const std::string& src, const char seq=' ');

/*
 * @desc: 返回char的长度，eg digit->1; chines->3
 */
int parse_char(const char *str);

/*
 * @desc: 处理一个字符串中的标点符号
 *
 * param[in]: 输入字符串
 * param[in|put]: 处理后的字符串
 * param[in]: 把标点符号替换成指定的字符，默认为空
 *
 * return -1:failed, 0:success
 */
int punct_process(const std::string &raw_str, std::string &norm_str, 
        const std::string &replacer="");

/*
 * @desc: 判断一个字符是不是中文标点
 */
bool is_cn_punct(const std::string &word);

} //end namesapce string_util
