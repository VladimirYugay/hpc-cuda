#include "aux.h"
#include "kernels/kernels.h"
#include "test_aux.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace ::testing;

class KernelsTests : public ::testing::Test {
  public:
    KernelsTests() : numRows(9) {

        y.resize(numRows, 0.0f);
        x.resize(numRows, 0.0f);
        for (int i = 0; i < x.size(); ++i) {
            x[i] = static_cast<float>(i + 1);
        }

        cudaMalloc(&devX, numRows * sizeof(float));
        CHECK_ERR;
        cudaMemcpy(devX, x.data(), numRows * sizeof(float), cudaMemcpyHostToDevice);
        CHECK_ERR;

        cudaMalloc(&devY, numRows * sizeof(float));
        CHECK_ERR;
    }

    ~KernelsTests() {
        cudaFree(devX);
        CHECK_ERR;
        cudaFree(devY);
        CHECK_ERR;
    }

  protected:
    void SetUp() override {}

    float *devX{nullptr};
    float *devY{nullptr};

    std::vector<float> x{};
    std::vector<float> y{};
    size_t numRows{};
};


//--------------------------------------------------------------------------------------------------
TEST_F(KernelsTests, CsrMvKernel) {
    CsrMatrix hostMatrix = get1DStencilCsrMatrix(numRows);
    DevCsrMatrix devMatrix;
    devMatrix.numRows = hostMatrix.numRows;
    devMatrix.nnz = hostMatrix.nnz;

    cudaMalloc(&devMatrix.values, devMatrix.nnz * sizeof(float));
    CHECK_ERR;
    cudaMemcpy(devMatrix.values, hostMatrix.values.data(), devMatrix.nnz * sizeof(float),
               cudaMemcpyHostToDevice);
    CHECK_ERR;

    cudaMalloc(&devMatrix.indices, devMatrix.nnz * sizeof(int));
    CHECK_ERR;
    cudaMemcpy(devMatrix.indices, hostMatrix.indices.data(), devMatrix.nnz * sizeof(int),
               cudaMemcpyHostToDevice);
    CHECK_ERR;

    cudaMalloc(&devMatrix.start, (devMatrix.numRows + 1) * sizeof(int));
    CHECK_ERR;
    cudaMemcpy(devMatrix.start, hostMatrix.start.data(), (devMatrix.numRows + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    CHECK_ERR;


    launch_csrMatVecMult(devY, devMatrix, devX, ExecutionMode::PAGERANK);
    CHECK_ERR;

    std::vector<float> results(hostMatrix.numRows, 0.0f);
    cudaMemcpy(const_cast<float *>(results.data()), devY, hostMatrix.numRows * sizeof(float),
               cudaMemcpyDeviceToHost);
    CHECK_ERR;


    std::vector<float> expectedResult{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -10.0};
    EXPECT_THAT(results, ElementsAreArray(getExpectedArray(expectedResult)));

    cudaFree(devMatrix.values);
    CHECK_ERR;
    cudaFree(devMatrix.indices);
    CHECK_ERR;
    cudaFree(devMatrix.start);
    CHECK_ERR;
}


//--------------------------------------------------------------------------------------------------
TEST_F(KernelsTests, CsrMvKernelVectorized) {
  CsrMatrix hostMatrix = get1DStencilCsrMatrix(numRows);
  DevCsrMatrix devMatrix;
  devMatrix.numRows = hostMatrix.numRows;
  devMatrix.nnz = hostMatrix.nnz;

  cudaMalloc(&devMatrix.values, devMatrix.nnz * sizeof(float));
  CHECK_ERR;
  cudaMemcpy(devMatrix.values, hostMatrix.values.data(), devMatrix.nnz * sizeof(float),
             cudaMemcpyHostToDevice);
  CHECK_ERR;

  cudaMalloc(&devMatrix.indices, devMatrix.nnz * sizeof(int));
  CHECK_ERR;
  cudaMemcpy(devMatrix.indices, hostMatrix.indices.data(), devMatrix.nnz * sizeof(int),
             cudaMemcpyHostToDevice);
  CHECK_ERR;

  cudaMalloc(&devMatrix.start, (devMatrix.numRows + 1) * sizeof(int));
  CHECK_ERR;
  cudaMemcpy(devMatrix.start, hostMatrix.start.data(), (devMatrix.numRows + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  CHECK_ERR;


  launch_csrMatVecMult(devY, devMatrix, devX, ExecutionMode::PAGERANK_VECTORIZED);
  CHECK_ERR;

  std::vector<float> results(hostMatrix.numRows, 0.0f);
  cudaMemcpy(const_cast<float *>(results.data()), devY, hostMatrix.numRows * sizeof(float),
             cudaMemcpyDeviceToHost);
  CHECK_ERR;


  std::vector<float> expectedResult{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -10.0};
  EXPECT_THAT(results, ElementsAreArray(getExpectedArray(expectedResult)));

  cudaFree(devMatrix.values);
  CHECK_ERR;
  cudaFree(devMatrix.indices);
  CHECK_ERR;
  cudaFree(devMatrix.start);
  CHECK_ERR;
}


//--------------------------------------------------------------------------------------------------
TEST_F(KernelsTests, EllMvKernel) {
    EllMatrix hostMatrix;
    hostMatrix.numRows = numRows;
    hostMatrix.numColsPerRow = 4;

    // clang-format off
    std::vector<float> values{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0};

    std::vector<int> indices {4, 6, 7, 1, 0, 6, 1, 5, 0,
                              0, 0, 0, 0, 1, 0, 0, 8, 0,
                              0, 0, 0, 0, 2, 0, 0, 0, 0,
                              0, 0, 0, 0, 3, 0, 0, 0, 0};
    // clang-format on

    DevEllMatrix devMatrix;
    devMatrix.numRows = hostMatrix.numRows;
    devMatrix.numColsPerRow = hostMatrix.numColsPerRow;

    const size_t memRequired = hostMatrix.numColsPerRow * hostMatrix.numRows;
    cudaMalloc(&devMatrix.values, memRequired * sizeof(float));
    CHECK_ERR;
    cudaMemcpy(devMatrix.values, values.data(), memRequired * sizeof(float),
               cudaMemcpyHostToDevice);
    CHECK_ERR;

    cudaMalloc(&devMatrix.indices, memRequired * sizeof(int));
    CHECK_ERR;
    cudaMemcpy(devMatrix.indices, indices.data(), memRequired * sizeof(int),
               cudaMemcpyHostToDevice);
    CHECK_ERR;

    launch_ellMatVecMult(devY, devMatrix, devX);
    CHECK_ERR;

    std::vector<float> results(hostMatrix.numRows, 0.0f);
    cudaMemcpy(const_cast<float *>(results.data()), devY, hostMatrix.numRows * sizeof(float),
               cudaMemcpyDeviceToHost);
    CHECK_ERR;


    std::vector<float> expectedResult{5.0, 7.0, 8.0, 2.0, 10.0, 7.0, 2.0, 15.0, 0.0};
    EXPECT_THAT(results, ElementsAreArray(getExpectedArray(expectedResult)));

    cudaFree(devMatrix.values);
    CHECK_ERR;
    cudaFree(devMatrix.indices);
    CHECK_ERR;
}


//--------------------------------------------------------------------------------------------------
TEST_F(KernelsTests, BandMvKernel) {
    EllMatrix hostMatrix;
    hostMatrix.numRows = numRows;
    hostMatrix.numColsPerRow = 3;

    // clang-format off
    std::vector<float> values{0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                              2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                              -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0};
    // clang-format on

    DevBandMatrix devMatrix;
    devMatrix.numRows = hostMatrix.numRows;
    devMatrix.halfSize = 1;

    const size_t memRequired = hostMatrix.numColsPerRow * hostMatrix.numRows;
    cudaMalloc(&devMatrix.values, memRequired * sizeof(float));
    CHECK_ERR;
    cudaMemcpy(devMatrix.values, values.data(), memRequired * sizeof(float),
               cudaMemcpyHostToDevice);
    CHECK_ERR;

    launch_bandMatVecMult(devY, devMatrix, devX);
    CHECK_ERR;

    std::vector<float> results(hostMatrix.numRows, 0.0f);
    cudaMemcpy(const_cast<float *>(results.data()), devY, hostMatrix.numRows * sizeof(float),
               cudaMemcpyDeviceToHost);
    CHECK_ERR;


    std::vector<float> expectedResult{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0};
    EXPECT_THAT(results, ElementsAreArray(getExpectedArray(expectedResult)));

    cudaFree(devMatrix.values);
    CHECK_ERR;
}