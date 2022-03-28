#include "aux.h"
#include "test_aux.h"
#include "util.h"
#include <string>

using namespace ::testing;

TEST(MatrixGeneration, EllFormat) {
    const size_t numRows = 6;
    EllMatrix testMatrix = get1DStencilEllMatrix(numRows);

    EXPECT_EQ(testMatrix.numRows, numRows);
    EXPECT_EQ(testMatrix.numColsPerRow, 3);
    EXPECT_EQ(testMatrix.values.size(), numRows * 3);
    EXPECT_EQ(testMatrix.indices.size(), numRows * 3);

    std::vector<float> expectedValues{0.0,  1.0,  1.0,  1.0, 1.0, 1.0, -2.0, -2.0, -2.0,
                                      -2.0, -2.0, -2.0, 1.0, 1.0, 1.0, 1.0,  1.0,  0.0};

    EXPECT_THAT(testMatrix.values, ElementsAreArray(getExpectedArray(expectedValues)));

    std::vector<int> expectedIndices{0, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 0};

    EXPECT_THAT(testMatrix.indices, ElementsAreArray(expectedIndices));
}


TEST(MatrixGeneration, CsrFormat) {
    const size_t numRows = 6;
    CsrMatrix testMatrix = get1DStencilCsrMatrix(numRows);

    EXPECT_EQ(testMatrix.numRows, numRows);
    EXPECT_EQ(testMatrix.nnz, numRows * 3);

    EXPECT_EQ(testMatrix.start.size(), numRows + 1);
    std::vector<int> expectedStart{0, 3, 6, 9, 12, 15, 18};
    EXPECT_THAT(testMatrix.start, ElementsAreArray(expectedStart));

    EXPECT_EQ(testMatrix.values.size(), numRows * 3);
    std::vector<float> expectedValues{0.0, -2.0, 1.0, 1.0, -2.0, 1.0, 1.0, -2.0, 1.0,
                                      1.0, -2.0, 1.0, 1.0, -2.0, 1.0, 1.0, -2.0, 0.0};
    EXPECT_THAT(testMatrix.values, ElementsAreArray(getExpectedArray(expectedValues)));

    EXPECT_EQ(testMatrix.indices.size(), numRows * 3);
    std::vector<int> expectedIndicies{0, 0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 0};
    EXPECT_THAT(testMatrix.indices, ElementsAreArray(expectedIndicies));
}


TEST(MatrixGeneration, LoadScsMatrix) {
    std::string fileName{"./test_matrix.mtx"};
    CsrMatrix testMatrix = loadMarketMatrix(fileName);
    const size_t expectedNumRows = 9;
    const size_t expectedNnz = 12;

    EXPECT_EQ(testMatrix.numRows, expectedNumRows);
    EXPECT_EQ(testMatrix.nnz, expectedNnz);

    std::vector<float> expectedValues(expectedNnz, 1.0);
    EXPECT_THAT(testMatrix.values, ElementsAreArray(getExpectedArray(expectedValues)));

    EXPECT_EQ(testMatrix.indices.size(), expectedNnz);
    std::vector<int> expectedIndicies{4, 6, 7, 1, 0, 1, 2, 3, 6, 1, 5, 8};
    EXPECT_THAT(testMatrix.indices, ElementsAreArray(expectedIndicies));

    EXPECT_EQ(testMatrix.start.size(), expectedNumRows + 1);
    std::vector<int> expectedStart{0, 1, 2, 3, 4, 8, 9, 10, 12, 12};
    EXPECT_THAT(testMatrix.start, ElementsAreArray(expectedStart));
}