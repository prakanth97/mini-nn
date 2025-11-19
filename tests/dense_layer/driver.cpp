#include <iostream>
#include <vector>
#include <random>
#include <cstdint>

//===----------------------------------------------------------------------===//
// MemRef descriptors for rank-1 and rank-2 tensors
//===----------------------------------------------------------------------===//
struct MemRef1D {
  float* allocated;
  float* aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
};

struct MemRef2D {
  float* allocated;
  float* aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
};

//===----------------------------------------------------------------------===//
// extern "C" declaration — must match lowered LLVM IR exactly
//===----------------------------------------------------------------------===//
extern "C" void model(
    // input (1x10)
    float* input_alloc, float* input_aligned, int64_t input_offset,
    int64_t input_dim0, int64_t input_dim1, int64_t input_stride0, int64_t input_stride1,

    // w1 (10x5)
    float* w1_alloc, float* w1_aligned, int64_t w1_offset,
    int64_t w1_dim0, int64_t w1_dim1, int64_t w1_stride0, int64_t w1_stride1,

    // b1 (5)
    float* b1_alloc, float* b1_aligned, int64_t b1_offset,
    int64_t b1_dim0, int64_t b1_stride0,

    // w2 (5x1)
    float* w2_alloc, float* w2_aligned, int64_t w2_offset,
    int64_t w2_dim0, int64_t w2_dim1, int64_t w2_stride0, int64_t w2_stride1,

    // b2 (1)
    float* b2_alloc, float* b2_aligned, int64_t b2_offset,
    int64_t b2_dim0, int64_t b2_stride0,

    // output tensor (1x1) - passed by reference
    float* out_alloc, float* out_aligned, int64_t out_offset,
    int64_t out_dim0, int64_t out_dim1, int64_t out_stride0, int64_t out_stride1
  );


//===----------------------------------------------------------------------===//
// Utility helpers
//===----------------------------------------------------------------------===//

// Fill random floats in [-1, 1]
void fillRandom(std::vector<float>& v) {
  static std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& x : v) x = dist(gen);
}

// Print a 2D MemRef
void printMemRef(const MemRef2D& memref) {
  std::cout << "Result tensor (" << memref.sizes[0] << "x" << memref.sizes[1] << "): ";
  for (int64_t i = 0; i < memref.sizes[0] * memref.sizes[1]; ++i) {
    if (i > 0) std::cout << ", ";
    std::cout << memref.aligned[i];
  }
  std::cout << std::endl;
}

float relu(float x) {
  return x > 0.0f ? x : 0.0f;
}

float sigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

float inference(
  const std::vector<float>& input,
  const std::vector<float>& w1,
  const std::vector<float>& b1,
  const std::vector<float>& w2,
  const std::vector<float>& b2
) {
  // First layer
  std::vector hidden(5, 0.0f);
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 10; ++j) {
      hidden[i] += input[j] * w1[j* 5 + i];
    }
    hidden[i] += b1[i];
    hidden[i] = relu(hidden[i]);
  }

  // Second layer
  float output = 0.0f;
  for (int i = 0; i < 5; ++i) {
    output += hidden[i] * w2[i];
  }
  output += b2[0];
  output = sigmoid(output);
  return output;
}


//===----------------------------------------------------------------------===//
// main
//===----------------------------------------------------------------------===//
int main() {
  // Tensor shapes
  const int64_t input_dims[2] = {1, 10};
  const int64_t w1_dims[2]    = {10, 5};
  const int64_t b1_dims[1]    = {5};
  const int64_t w2_dims[2]    = {5, 1};
  const int64_t b2_dims[1]    = {1};
  const int64_t output_dims[2] = {1, 1};

  // Allocate storage
  std::vector<float> input(1 * 10);
  std::vector<float> w1(10 * 5);
  std::vector<float> b1(5);
  std::vector<float> w2(5 * 1);
  std::vector<float> b2(1);
  std::vector<float> output(1 * 1);

  // Initialize with random values
  fillRandom(input);
  fillRandom(w1);
  fillRandom(b1);
  fillRandom(w2);
  fillRandom(b2);

  // Compute row-major strides
  int64_t input_strides[2] = {input_dims[1], 1};
  int64_t w1_strides[2]    = {w1_dims[1], 1};
  int64_t b1_strides[1]    = {1};
  int64_t w2_strides[2]    = {w2_dims[1], 1};
  int64_t b2_strides[1]    = {1};
  int64_t output_strides[2] = {output_dims[1], 1};

  std::cout << "Running model..." << std::endl;

  // Call the MLIR-lowered function
  model(
      input.data(), input.data(), 0,
      input_dims[0], input_dims[1], input_strides[0], input_strides[1],
      w1.data(), w1.data(), 0,
      w1_dims[0], w1_dims[1], w1_strides[0], w1_strides[1],
      b1.data(), b1.data(), 0,
      b1_dims[0], b1_strides[0],
      w2.data(), w2.data(), 0,
      w2_dims[0], w2_dims[1], w2_strides[0], w2_strides[1],
      b2.data(), b2.data(), 0,
      b2_dims[0], b2_strides[0],
      output.data(), output.data(), 0,
      output_dims[0], output_dims[1], output_strides[0], output_strides[1]
  );

  std::cout << "Model executed successfully.\n";
  
  // Create a MemRef2D structure for printing
  MemRef2D result = {
    output.data(), output.data(), 0,
    {output_dims[0], output_dims[1]},
    {output_strides[0], output_strides[1]}
  };
  printMemRef(result);
  
  // Validate the output
  float actualVal = inference(input, w1, b1, w2, b2);
  float diff = std::abs(actualVal - output[0]);
  std::cout << "Validation: expected " << actualVal << ", got " << output[0]
            << ", diff = " << diff << std::endl;
  
  return 0;
}
