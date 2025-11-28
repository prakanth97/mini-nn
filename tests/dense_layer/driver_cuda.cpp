#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h> // Required for cudaMallocManaged, cudaDeviceSynchronize
#include <iostream>
#include <random>

//===----------------------------------------------------------------------===//
// MemRef descriptors
//===----------------------------------------------------------------------===//
struct MemRef2D {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
};

//===----------------------------------------------------------------------===//
// extern "C" declaration — matches LLVM IR
//===----------------------------------------------------------------------===//
extern "C" void
model(float *input_alloc, float *input_aligned, int64_t input_offset,
      int64_t input_dim0, int64_t input_dim1, int64_t input_stride0,
      int64_t input_stride1, float *w1_alloc, float *w1_aligned,
      int64_t w1_offset, int64_t w1_dim0, int64_t w1_dim1, int64_t w1_stride0,
      int64_t w1_stride1, float *b1_alloc, float *b1_aligned, int64_t b1_offset,
      int64_t b1_dim0, int64_t b1_stride0, float *w2_alloc, float *w2_aligned,
      int64_t w2_offset, int64_t w2_dim0, int64_t w2_dim1, int64_t w2_stride0,
      int64_t w2_stride1, float *b2_alloc, float *b2_aligned, int64_t b2_offset,
      int64_t b2_dim0, int64_t b2_stride0, float *out_alloc, float *out_aligned,
      int64_t out_offset, int64_t out_dim0, int64_t out_dim1,
      int64_t out_stride0, int64_t out_stride1);

//===----------------------------------------------------------------------===//
// Utility helpers
//===----------------------------------------------------------------------===//

// Helper to allocate Unified Memory (accessible by CPU and GPU)
float *allocateManaged(size_t num_elements) {
  float *ptr;
  cudaError_t err = cudaMallocManaged(&ptr, num_elements * sizeof(float));
  if (err != cudaSuccess) {
    std::cerr << "CUDA Malloc Failed: " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
  return ptr;
}

// Fill random floats in [-1, 1] - Modified to take raw pointer
void fillRandom(float *ptr, size_t size) {
  static std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < size; ++i) {
    ptr[i] = dist(gen);
  }
}

void printMemRef(float *data, int64_t rows, int64_t cols) {
  std::cout << "Result tensor (" << rows << "x" << cols << "): ";
  for (int64_t i = 0; i < rows * cols; ++i) {
    if (i > 0)
      std::cout << ", ";
    std::cout << data[i];
  }
  std::cout << std::endl;
}

extern "C" void *my_managed_malloc(size_t size) {
  void *ptr;
  // Force all internal IR allocations to use Unified Memory
  cudaError_t err = cudaMallocManaged(&ptr, size);
  if (err != cudaSuccess) {
    std::cerr << "Managed Malloc Failed: " << cudaGetErrorString(err)
              << std::endl;
    exit(1);
  }
  return ptr;
}

extern "C" void my_managed_free(void *ptr) { cudaFree(ptr); }

float relu(float x) { return x > 0.0f ? x : 0.0f; }

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

float inference(
  float* input,
  float* w1,
  float* b1,
  float* w2,
  float* b2
) {
  // First layer
  float hidden[5] = {0.0f};
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 10; ++j) {
      hidden[i] += input[j] * w1[j * 5 + i];
    }
    hidden[i] += b1[i];
    hidden[i] = relu(hidden[i]);
  }

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
  // 1. Define Shapes
  const int64_t input_dims[2] = {1, 10};
  const int64_t w1_dims[2] = {10, 5};
  const int64_t b1_dims[1] = {5};
  const int64_t w2_dims[2] = {5, 1};
  const int64_t b2_dims[1] = {1};
  const int64_t output_dims[2] = {1, 1};

  // 2. Allocate Unified Memory (Managed)
  // These pointers work on Host (CPU) and Device (GPU)
  float *m_input = allocateManaged(1 * 10);
  float *m_w1 = allocateManaged(10 * 5);
  float *m_b1 = allocateManaged(5);
  float *m_w2 = allocateManaged(5 * 1);
  float *m_b2 = allocateManaged(1);
  float *m_output = allocateManaged(1 * 1);

  // 3. Initialize Data (CPU writes to Unified Memory)
  fillRandom(m_input, 10);
  fillRandom(m_w1, 50);
  fillRandom(m_b1, 5);
  fillRandom(m_w2, 5);
  fillRandom(m_b2, 1);
  // Initialize output to 0
  m_output[0] = 0.0f;

  // 4. Compute Strides
  int64_t input_strides[2] = {input_dims[1], 1};
  int64_t w1_strides[2] = {w1_dims[1], 1};
  int64_t b1_strides[1] = {1};
  int64_t w2_strides[2] = {w2_dims[1], 1};
  int64_t b2_strides[1] = {1};
  int64_t output_strides[2] = {output_dims[1], 1};

  std::cout << "Calling MLIR model function..." << std::endl;

  // 5. Call Model
  // We pass the managed pointers. The GPU driver handles the migration.
  model(m_input, m_input, 0, input_dims[0], input_dims[1], input_strides[0],
        input_strides[1], m_w1, m_w1, 0, w1_dims[0], w1_dims[1], w1_strides[0],
        w1_strides[1], m_b1, m_b1, 0, b1_dims[0], b1_strides[0], m_w2, m_w2, 0,
        w2_dims[0], w2_dims[1], w2_strides[0], w2_strides[1], m_b2, m_b2, 0,
        b2_dims[0], b2_strides[0], m_output, m_output, 0, output_dims[0],
        output_dims[1], output_strides[0], output_strides[1]);

  // 6. Synchronize
  // Wait for GPU to finish before CPU reads m_output
  cudaDeviceSynchronize();

  std::cout << "Model executed successfully." << std::endl;

  // 7. Read Result (CPU reads from Unified Memory)
  printMemRef(m_output, output_dims[0], output_dims[1]);

  // Valiadate with CPU inference
  float actualVal = inference(m_input, m_w1, m_b1, m_w2, m_b2);
  float diff = std::abs(actualVal - m_output[0]);
  std::cout << "Validation: expected " << actualVal << ", got" << m_output[0] 
      << ", diff =" << diff << std::endl;

  // 8. Cleanup
  cudaFree(m_input);
  cudaFree(m_w1);
  cudaFree(m_b1);
  cudaFree(m_w2);
  cudaFree(m_b2);
  cudaFree(m_output);

  return 0;
}
