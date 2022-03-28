#include "aux.h"
#include <algorithm>
#include <cstdlib>
#include <vector>


// A single matrix element
struct MatrixElement {
    int i{};
    int j{};
    float value{};
};


void convertCooToCsr(CsrMatrix &matrix, const std::vector<int>& I) {
    std::vector<MatrixElement> matrixElements(matrix.nnz);

    // use additional structure for sorting
    for (int x = 0; x < matrix.nnz; ++x) {
        matrixElements[x].i = I[x];
        matrixElements[x].j = matrix.indices[x];
        matrixElements[x].value = matrix.values[x];
    }

    // sort after collumn and then after i
    //std::qsort(m, nnz, sizeof(mat_elem), cmp_by_ij);
    std::sort(matrixElements.begin(), matrixElements.end(),
              [](MatrixElement &first, MatrixElement &second) -> bool {
                  if (first.i > second.i) {
                      return false;
                  } else if (first.i < second.i) {
                      return true;
                  } else {
                      if (first.j > second.j) {
                          return false;
                      } else if (first.j < second.j) {
                          return true;
                      } else {
                          return false;
                      }
                  }
              });

    // set all ptr to 0
    for (int x = 0; x < (matrix.numRows + 1); ++x) {
        matrix.start[x] = -1;
    }

    // fill in ptr
    int currRow = 0;
    matrix.start[currRow] = 0;
    for (int x = 0; x < matrix.nnz; ++x) {
        if (matrixElements[x].i > currRow) {
            currRow = matrixElements[x].i;
            matrix.start[currRow] = x;
        }

        matrix.indices[x] = matrixElements[x].j;
        matrix.values[x] = matrixElements[x].value;
    }

    // last element in ptr is set to nnz
    matrix.start[matrix.numRows] = matrix.nnz;

    // all missing rows get the ptr of the previous row
    for (int x = (matrix.numRows - 1); x >= 0; --x) {
        if (matrix.start[x] == -1) {
            matrix.start[x] = matrix.start[x + 1];
        }
    }
}


EllMatrix get1DStencilEllMatrix(const int numPoints) {
  constexpr int NUM_COLS_PER_ROW = 3;
  EllMatrix matrix(numPoints, NUM_COLS_PER_ROW);

  // fill matrix with stencil [1 -2 1]
  for (int i = 1; i < (numPoints - 1); ++i) {
    matrix.indices[i] = i - 1;
    matrix.indices[i + numPoints * 1] = i;
    matrix.indices[i + numPoints * 2] = i + 1;

    matrix.values[i] = 1.0f;
    matrix.values[i + numPoints * 1] = -2.0f;
    matrix.values[i + numPoints * 2] = 1.0f;
  }

  // first and last line (Dirichlet condition)
  matrix.indices[0 + numPoints * 1] = 0;
  matrix.indices[0 + numPoints * 2] = 1;
  matrix.values[0 + numPoints * 1] = -2.0f;
  matrix.values[0 + numPoints * 2] = 1.0f;

  matrix.indices[(numPoints - 1) + numPoints * 0] = numPoints - 2;
  matrix.indices[(numPoints - 1) + numPoints * 1] = numPoints - 1;
  matrix.values[(numPoints - 1) + numPoints * 0] = 1.0f;
  matrix.values[(numPoints - 1) + numPoints * 1] = -2.0f;
  return matrix;
}


CsrMatrix get1DStencilCsrMatrix(const int numPoints) {
  CsrMatrix matrix;

  constexpr int NUM_COLS_PER_ROW = 3;
  matrix.numRows = numPoints;
  matrix.nnz = NUM_COLS_PER_ROW * matrix.numRows;

  matrix.start.resize(numPoints + 1, 0);
  for (int i = 0; i <= numPoints; ++i) {
    matrix.start[i] = 3 * i;
  }

  matrix.values.resize(matrix.nnz, 0.0f);
  matrix.indices.resize(matrix.nnz, 0);

  // fill matrix with stencil [1 -2 1]
  for (int i = 1; i < (numPoints - 1); ++i) {
    matrix.indices[NUM_COLS_PER_ROW * i] = i - 1;
    matrix.indices[NUM_COLS_PER_ROW * i + 1] = i;
    matrix.indices[NUM_COLS_PER_ROW * i + 2] = i + 1;

    matrix.values[NUM_COLS_PER_ROW * i] = 1;
    matrix.values[NUM_COLS_PER_ROW * i + 1] = -2;
    matrix.values[NUM_COLS_PER_ROW * i + 2] = 1;
  }

  // first and last line (Dirichlet condition)
  matrix.indices[NUM_COLS_PER_ROW * 0 + 1] = 0;
  matrix.indices[NUM_COLS_PER_ROW * 0 + 2] = 1;
  matrix.values[NUM_COLS_PER_ROW * 0 + 1] = -2;
  matrix.values[NUM_COLS_PER_ROW * 0 + 2] = 1;

  matrix.indices[NUM_COLS_PER_ROW * (numPoints - 1)] = numPoints - 2;
  matrix.indices[NUM_COLS_PER_ROW * (numPoints - 1) + 1] = numPoints - 1;
  matrix.values[NUM_COLS_PER_ROW * (numPoints - 1)] = 1;
  matrix.values[NUM_COLS_PER_ROW * (numPoints - 1) + 1] = -2;

  return matrix;
}