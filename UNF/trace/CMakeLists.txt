cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(predict predict.cc lib/tokenizer.cc lib/string_util.cc)
target_include_directories(predict PUBLIC "lib")
target_link_libraries(predict "${TORCH_LIBRARIES}")
set_property(TARGET predict PROPERTY CXX_STANDARD 11)
