#include "aux.h"
#include "ht.h"
#include "kernels/kernels.h"


namespace ht {
void PoissonSolver::poisson_cusparse() {

    const size_t N = numGridPoints;
    CsrMatrix matrix = get1DStencilCsrMatrix(N);

    DevCsrMatrix devMatrix;
    devMatrix.nnz = matrix.nnz;
    devMatrix.numRows = matrix.numRows;

    // clang-format off
    float *devX{nullptr};
    cudaMalloc(&devX, N * sizeof(float)); CHECK_ERR;
    cudaMemcpy(devX, x.data(), N * sizeof(float), cudaMemcpyHostToDevice); CHECK_ERR;

    float *devY{nullptr};
    cudaMalloc(&devY, N * sizeof(float)); CHECK_ERR;
    cudaMemcpy(devY, y.data(), N * sizeof(float), cudaMemcpyHostToDevice); CHECK_ERR;

    cudaMalloc(&devMatrix.start, matrix.start.size() * sizeof(int)); CHECK_ERR;
    cudaMemcpy(devMatrix.start, matrix.start.data(), matrix.start.size() * sizeof(int),
               cudaMemcpyHostToDevice); CHECK_ERR;

    cudaMalloc(&devMatrix.indices, matrix.indices.size() * sizeof(int)); CHECK_ERR;
    cudaMemcpy(devMatrix.indices, matrix.indices.data(), matrix.indices.size() * sizeof(int),
               cudaMemcpyHostToDevice); CHECK_ERR;

    cudaMalloc(&devMatrix.values, matrix.values.size() * sizeof(float)); CHECK_ERR;
    cudaMemcpy(devMatrix.values, matrix.values.data(), matrix.values.size() * sizeof(float),
               cudaMemcpyHostToDevice); CHECK_ERR;
    // clang-format on


    cublasHandle_t cublasHandle{nullptr};
    CUBLAS_CHECK(cublasCreate(&cublasHandle));

    cusparseHandle_t cusparseHandle{nullptr};
    CUSPARSE_CHECK(cusparseCreate(&cusparseHandle));

    cusparseMatDescr_t descr{nullptr};
    CUSPARSE_CHECK(cusparseCreateMatDescr(&descr));
    CUSPARSE_CHECK(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CHECK(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));

    /*
    // Note: deprecated in cuda-sdk@11
    cusparseHybMat_t hybridMatrix{nullptr};
    CUSPARSE_CHECK(cusparseCreateHybMat(&hybridMatrix));
    CUSPARSE_CHECK(cusparseScsr2hyb(cusparseHandle, devMatrix.numRows, devMatrix.numRows, descr,
                                    devMatrix.values, devMatrix.start, devMatrix.indices,
                                    hybridMatrix, 3, CUSPARSE_HYB_PARTITION_MAX));
    */

    const float dx = 1.0f / (float)(N - 1);
    const float dt = (0.5f * dx * dx) / settings.conductivity;
    float err = std::numeric_limits<float>::max();
    for (int counter = 0; ((err > settings.epsilon) && (settings.numIterations > counter)); ++counter) {
        // launch_ellMatVecMult(devY, devMatrix, devX);
        float alpha = 1.0, beta = 0.0;
        // TODO: H4.1
        //CUSPARSE_CHECK(cusparseShybmv(...));

        // computes the Euclidean norm of the vector devY using cublas
        // TODO: H4.1 similar to T4.1
        //CUBLAS_CHECK(cublas(...));

        alpha = dt / (dx * dx) * settings.conductivity;
        // TODO: H4.1 similar to T4.1
        //CUBLAS_CHECK(cublas(...));

        if ((counter % settings.stepsToPrint) == 0) {
            {
                std::lock_guard<std::mutex> guard{plottingMutex};
                cudaMemcpy(const_cast<float *>(x.data()), devX, N * sizeof(float),
                           cudaMemcpyDeviceToHost);
                CHECK_ERR;
                status = PlottingStatus::UPDATED;
            }

            std::cout << "time = " << dt * static_cast<float>(counter) << ", err = " << err
                      << ", Temperature at x = 0.5: " << x[N / 2] << std::endl;
        }
    }

    // clang-format off
    cudaFree(devX); CHECK_ERR;
    cudaFree(devY); CHECK_ERR;
    cudaFree(devMatrix.start); CHECK_ERR;
    cudaFree(devMatrix.indices); CHECK_ERR;
    cudaFree(devMatrix.values); CHECK_ERR;

    // Note: deprecated in cuda-sdk@11
    // CUSPARSE_CHECK(cusparseDestroyHybMat(hybridMatrix));
    CUSPARSE_CHECK(cusparseDestroy(cusparseHandle));
    CUBLAS_CHECK(cublasDestroy(cublasHandle));
    // clang-format on
}
} // namespace ht
