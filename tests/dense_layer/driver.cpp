#include <iostream>
#include <vector>
#include <random>
#include <cstdint>

// extern "C" linkage — matches your compiled symbol
extern "C" {
  void* model( // returns struct by value (can be handled differently depending on ABI)
      float* input_alloc, float* input_aligned, int64_t input_offset,
      int64_t input_dim0, int64_t input_dim1, int64_t input_stride0, int64_t input_stride1,
      float* w1_alloc, float* w1_aligned, int64_t w1_offset,
      int64_t w1_dim0, int64_t w1_dim1, int64_t w1_stride0, int64_t w1_stride1,
      float* b1_alloc, float* b1_aligned, int64_t b1_offset,
      int64_t b1_dim0, int64_t b1_stride0,
      float* w2_alloc, float* w2_aligned, int64_t w2_offset,
      int64_t w2_dim0, int64_t w2_dim1, int64_t w2_stride0, int64_t w2_stride1,
      float* b2_alloc, float* b2_aligned, int64_t b2_offset,
      int64_t b2_dim0, int64_t b2_stride0);
}

// Fill random floats in [-1, 1]
void fillRandom(std::vector<float>& v) {
  static std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& x : v) x = dist(gen);
}

int main() {
  // Define tensor shapes
  const int64_t input_dims[2] = {1, 10};
  const int64_t w1_dims[2] = {10, 5};
  const int64_t b1_dims[1] = {5};
  const int64_t w2_dims[2] = {5, 1};
  const int64_t b2_dims[1] = {1};

  // Allocate data
  std::vector<float> input(1 * 10);
  std::vector<float> w1(10 * 5);
  std::vector<float> b1(5);
  std::vector<float> w2(5 * 1);
  std::vector<float> b2(1);

  fillRandom(input);
  fillRandom(w1);
  fillRandom(b1);
  fillRandom(w2);
  fillRandom(b2);

  // Strides (row-major)
  int64_t input_strides[2] = {input_dims[1], 1};
  int64_t w1_strides[2] = {w1_dims[1], 1};
  int64_t b1_strides[1] = {1};
  int64_t w2_strides[2] = {w2_dims[1], 1};
  int64_t b2_strides[1] = {1};

  // Call function (result struct returned by value)
  void* result = model(
      input.data(), input.data(), 0, input_dims[0], input_dims[1], input_strides[0], input_strides[1],
      w1.data(), w1.data(), 0, w1_dims[0], w1_dims[1], w1_strides[0], w1_strides[1],
      b1.data(), b1.data(), 0, b1_dims[0], b1_strides[0],
      w2.data(), w2.data(), 0, w2_dims[0], w2_dims[1], w2_strides[0], w2_strides[1],
      b2.data(), b2.data(), 0, b2_dims[0], b2_strides[0]
  );

  // Depending on how your MLIR lowered return type, you may need to cast
  // and dereference to access the actual output tensor pointer.
  std::cout << "Model executed successfully (result struct at " << result << ")\n";
  return 0;
}
