#include <torch/script.h> 

#include<iostream>
#include<vector>
#include<string>

#include "tokenizer.h"


int max_seq_length = 8; //hard code, comfortable with the tracing process

int main(int argc, const char* argv[])
{
    if (argc != 3) {
        std::cerr << "usage predict <trace_model_path> <vocab_path>";
        return -1;
    }
    //step1 load model
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    //step2 init tokenizer
    Tokenizer tokenizer(argv[2]);
    uint32_t pad_index = tokenizer.get_pad_index();

    //step3 loop predict
    std::string text;
    while (true) 
    {
        std::cout << "\n" << "Input ->";
        getline(std::cin, text);
        if (text == "break") break;
        //segment
        std::vector<std::string> segment;
        tokenizer.tokenize(text, segment);

        //to id
        std::vector<float> t2id;
        tokenizer.token2id(segment, t2id);

        //padding
        int seg_len = segment.size();
        torch::Tensor mask = torch::ones({1, max_seq_length});

        while(seg_len < max_seq_length) {
            t2id.push_back(pad_index);
            mask[0][seg_len] = 0;
            seg_len ++;
        }

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(torch::from_blob(t2id.data(), {1, max_seq_length}).to(torch::kLong));
        inputs.push_back(mask.to(torch::kLong));
        std::cout << inputs[0] << std::endl;
        std::cout << inputs[1] << std::endl;
        torch::Tensor logits = module.forward(inputs).toTensor();
        //torch::Tensor logits = module.forward({input}).toTensor();
        std::cout << logits << std::endl;
    }
}
