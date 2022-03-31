#include "kernels/kernels.h"
#include "solver.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstdlib>
#include <limits>

using namespace ::testing;

TEST(Reduction, Algorithm) {
    const size_t testSize = 20000;
    Reducer reducer(testSize);

    std::vector<float> testVector(testSize, 1.0f);
    testVector[3 * testSize / 4 + 1] = 9.81f;
    float *devVector{nullptr};

    cudaMalloc(&devVector, testSize * sizeof(float));
    CHECK_ERR;

    cudaMemcpy(devVector, testVector.data(), testSize * sizeof(float), cudaMemcpyHostToDevice);
    CHECK_ERR;

    float testMax = reducer.reduceMax(devVector);
    EXPECT_EQ(testMax, 9.81f);

    cudaFree(devVector);
    CHECK_ERR;
}


class StencilTest : public ::testing::Test {
  public:
    StencilTest() {
        settings.num1dGridPoints = 1000;
        settings.numIterations = 1;
        settings.stepsPerPrint = 1;
        settings.blockSizeX = 32;
        settings.blockSizeY = 16;
        settings.eps = 0.0;
        settings.endTime = std::numeric_limits<float>::max();
        settings.mode = ExecutionMode::CPU;
        settings.isVerbose = false;

        const size_t numElements = settings.num1dGridPoints * settings.num1dGridPoints;
        initialField.resize(numElements, 1.0f);
        std::transform(initialField.begin(), initialField.end(), initialField.begin(),
                       [](float) { return static_cast<float>(rand() % 2); });

        Solver cpuSolver(settings);
        cpuSolver.init(initialField);
        cpuSolver.run();

        expectedSolution = cpuSolver.getSolution();
    }

  protected:
    Settings settings;
    std::vector<float> initialField;
    std::vector<float> expectedSolution;
};


std::vector<Matcher<float>> getExpectedArray(const std::vector<float> &expectedVector) {
    std::vector<Matcher<float>> expectedArray;
    for (const auto item : expectedVector) {
        expectedArray.emplace_back(item);
    }
    return expectedArray;
}


TEST_F(StencilTest, GLOBAL_MEM) {

    settings.mode = ExecutionMode::GPU_GLOB_MEM;
    Solver gpuSolver(settings);
    gpuSolver.init(initialField);
    gpuSolver.run();
    auto testSolution = gpuSolver.getSolution();

    EXPECT_THAT(testSolution, ElementsAreArray(getExpectedArray(expectedSolution)));
}


TEST_F(StencilTest, SHARED_MEM) {

    settings.mode = ExecutionMode::GPU_SHR_MEM;
    Solver gpuSolver(settings);
    gpuSolver.init(initialField);
    gpuSolver.run();
    auto testSolution = gpuSolver.getSolution();

    EXPECT_THAT(testSolution, ElementsAreArray(getExpectedArray(expectedSolution)));
}
