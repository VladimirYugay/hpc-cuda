#include "data_types.h"
#include "driver.h"
#include "kernels/kernels.h"
#include "util.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstdlib>
#include <vector>

using namespace ::testing;

std::vector<Matcher<float>> getExpectedArray(const std::vector<float> &Answer) {
    std::vector<Matcher<float>> expectedArray;
    for (const auto item : Answer) {
        expectedArray.emplace_back(item);
    }
    return expectedArray;
}

class MatMult : public ::testing::Test {
  protected:
    void SetUp() override {
        // prepare a fresh configuration for each test
        config = Configuration();
        config.numRepeats = 1;
        config.printInfo = false;
        config.printMatrix = false;
    }
    Configuration config{};
};


TEST_F(MatMult, CPU) {
    constexpr size_t matrixSize = 4;
    constexpr size_t totalSize = matrixSize * matrixSize;
    // col-major format 
    // std::vector<float> A = {1, 1, 0, 2, 0, 1, 3, 1, 1};
    // std::vector<float> B = {4, 0, 0, 0, 1, 0, 1, 2, 2};        
    std::vector<float> A = {1, 2, 1, 0, 2, 0, 1, 2, 0, 1, 2, 1, 1, 0, 0, 1};
    std::vector<float> B = {0, 1, 0, 1, 1, 0, 2, 0, 3, 3, 2, 2, 0, 3, 0, 1};            

    std::vector<float> TestC(totalSize, 0.0f);
    config.numRepeats = 1;
    config.tileSize = 2;
    config.matrixSize = matrixSize;
    config.kernelType = KernelType::KERNEL_CPU;
    cpu::matrixMult(TestC, A, B, config);
    std::vector<float> ExpectedC = {3, 0, 1, 3, 1, 4, 5, 2, 11, 8, 10, 10, 7, 0, 3, 7};

    auto expectedArray = getExpectedArray(ExpectedC);
    EXPECT_THAT(TestC, ElementsAreArray(expectedArray));
}


TEST(CudaGrids, Grid1D) {
    EXPECT_EQ(gpu::get1DGrid(4, 16), 4);
    EXPECT_EQ(gpu::get1DGrid(5, 16), 4);
    EXPECT_EQ(gpu::get1DGrid(8, 16), 2);
    EXPECT_EQ(gpu::get1DGrid(9, 16), 2);
    EXPECT_EQ(gpu::get1DGrid(16, 16), 1);
    EXPECT_EQ(gpu::get1DGrid(17, 16), 1);
}


struct TestMemory {
  public:
    TestMemory(size_t size) : matrixSize(size), totalSize(size * size) {
        TestC.resize(totalSize, 0.0);
        ExpectedC.resize(totalSize, 0.0);
        // A.resize(totalSize, 0.0);
        // B.resize(totalSize, 0.0);
    }

    std::vector<float> A{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<float> B{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    // std::vector<float> A{1, 2, 1, 0, 2, 0, 1, 2, 0, 1, 2, 1, 1, 0, 0, 1};
    // std::vector<float> B{0, 1, 0, 1, 1, 0, 2, 0, 3, 3, 2, 2, 0, 3, 0, 1};
    std::vector<float> TestC{};
    std::vector<float> ExpectedC{};

  private:
    size_t totalSize{};
    size_t matrixSize{};
};


// struct TestMemory {
//   public:
//     TestMemory(size_t size) : matrixSize(size), totalSize(size * size) {
//         TestC.resize(totalSize, 0.0);
//         ExpectedC.resize(totalSize, 0.0);
//         A.resize(totalSize, 0.0);
//         B.resize(totalSize, 0.0);

//         auto randomizer = [](float) { return static_cast<float>(rand() % 100); };
//         std::transform(A.begin(), A.end(), A.begin(), randomizer);
//         std::transform(B.begin(), B.end(), B.begin(), randomizer);
//     }

//     std::vector<float> A{};
//     std::vector<float> B{};
//     std::vector<float> TestC{};
//     std::vector<float> ExpectedC{};

//   private:
//     size_t totalSize{};
//     size_t matrixSize{};
// };


namespace test {
  namespace local {
    size_t size = 4;
  }
}


TEST_F(MatMult, GPU_CUBLAS) {
    config.tileSize = 4;
    config.matrixSize = test::local::size;
    config.kernelType = KernelType::KERNEL_CUBLAS;
    TestMemory memory(config.matrixSize);

    // run on GPU
    compute(memory.TestC, memory.A, memory.B, config);

    // run on CPU
    cpu::matrixMult(memory.ExpectedC, memory.A, memory.B, config);

    auto expectedArray = getExpectedArray(memory.ExpectedC);

    EXPECT_THAT(memory.TestC, ElementsAreArray(expectedArray));
}


TEST_F(MatMult, GPU_GLOBAL) {
    config.tileSize = 4;
    config.matrixSize = test::local::size;
    config.kernelType = KernelType::KERNEL_GLOBAL;
    TestMemory memory(config.matrixSize);

    // run on GPU
    compute(memory.TestC, memory.A, memory.B, config);

    // run on CPU
    cpu::matrixMult(memory.ExpectedC, memory.A, memory.B, config);

    auto expectedArray = getExpectedArray(memory.ExpectedC);

    EXPECT_THAT(memory.TestC, ElementsAreArray(expectedArray));
}


TEST_F(MatMult, GPU_TILED) {
    config.tileSize = 2;
    config.matrixSize = 4;
    config.kernelType = KernelType::KERNEL_TILED;
    TestMemory memory(config.matrixSize);

    // run on GPU
    compute(memory.TestC, memory.A, memory.B, config);

    // run on CPU
    cpu::matrixMult(memory.ExpectedC, memory.A, memory.B, config);

    auto expectedArray = getExpectedArray(memory.ExpectedC);

    EXPECT_THAT(memory.TestC, ElementsAreArray(expectedArray));
}


TEST_F(MatMult, GPU_COALESCED) {
    config.tileSize = 4;
    config.matrixSize = test::local::size;
    config.kernelType = KernelType::KERNEL_COALESCED;
    TestMemory memory(config.matrixSize);

    // run on GPU
    compute(memory.TestC, memory.A, memory.B, config);

    // run on CPU
    cpu::matrixMult(memory.ExpectedC, memory.A, memory.B, config);

    auto expectedArray = getExpectedArray(memory.ExpectedC);

    EXPECT_THAT(memory.TestC, ElementsAreArray(expectedArray));
}


TEST_F(MatMult, GPU_COALESCED_DYM) {
    config.tileSize = 4;
    config.matrixSize = test::local::size;
    config.kernelType = KernelType::KERNEL_COALESCED_DYM;
    TestMemory memory(config.matrixSize);

    // run on GPU
    compute(memory.TestC, memory.A, memory.B, config);

    // run on CPU
    cpu::matrixMult(memory.ExpectedC, memory.A, memory.B, config);

    auto expectedArray = getExpectedArray(memory.ExpectedC);

    EXPECT_THAT(memory.TestC, ElementsAreArray(expectedArray));
}


TEST_F(MatMult, GPU_OVERLAPPED) {
    config.tileSize = 4;
    config.matrixSize = test::local::size;
    config.kernelType = KernelType::KERNEL_OVERLAPPED;
    TestMemory memory(config.matrixSize);

    // run on GPU
    compute(memory.TestC, memory.A, memory.B, config);

    // run on CPU
    cpu::matrixMult(memory.ExpectedC, memory.A, memory.B, config);

    auto expectedArray = getExpectedArray(memory.ExpectedC);

    EXPECT_THAT(memory.TestC, ElementsAreArray(expectedArray));
}