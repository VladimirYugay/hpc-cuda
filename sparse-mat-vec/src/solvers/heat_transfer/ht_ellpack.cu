#include "cublas_v2.h"
#include "ht.h"
#include "data_types.h"
#include "aux.h"
#include "kernels/kernels.h"


namespace ht {

void PoissonSolver::poisson_ellpack() {

    const size_t N = numGridPoints;
    // clang-format off
    float *devX{nullptr};
    cudaMalloc(&devX, N * sizeof(float)); CHECK_ERR;
    cudaMemcpy(devX, x.data(), N * sizeof(float), cudaMemcpyHostToDevice); CHECK_ERR;

    float *devY{nullptr};
    cudaMalloc(&devY, N * sizeof(float)); CHECK_ERR;
    cudaMemcpy(devY, y.data(), N * sizeof(float), cudaMemcpyHostToDevice); CHECK_ERR;

    EllMatrix matrix = get1DStencilEllMatrix(N);
    DevEllMatrix devMatrix;
    devMatrix.numColsPerRow = matrix.numColsPerRow;
    devMatrix.numRows = matrix.numRows;

    cudaMalloc(&devMatrix.indices, N * matrix.numColsPerRow * sizeof(int)); CHECK_ERR;
    cudaMemcpy(devMatrix.indices, matrix.indices.data(), N * matrix.numColsPerRow * sizeof(int),
               cudaMemcpyHostToDevice); CHECK_ERR;


    cudaMalloc(&devMatrix.values, N * matrix.numColsPerRow * sizeof(float)); CHECK_ERR;
    cudaMemcpy(devMatrix.values, matrix.values.data(), N * matrix.numColsPerRow * sizeof(float),
               cudaMemcpyHostToDevice); CHECK_ERR;
    // clang-format on


    /* Get handle to the CUBLAS context */
    cublasHandle_t cublasHandle = nullptr;
    CUBLAS_CHECK(cublasCreate(&cublasHandle));

    const float dx = 1.0f / (float)(N - 1);
    const float dt = (0.5f * dx * dx) / settings.conductivity;
    float err = std::numeric_limits<float>::max();
    CUBLAS_CHECK(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST));

    for (int counter = 0; ((err > settings.epsilon) && (settings.numIterations > counter)); ++counter) {
        launch_ellMatVecMult(devY, devMatrix, devX);

        // computes the Euclidean norm of the vector devY using cublas
        // TODO: T4.1 cuBLAS
        //CUBLAS_CHECK(cublas(...));

        float alpha = settings.conductivity * dt / (dx * dx);
      // TODO: T4.1: cuBLAS
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

    {
        std::lock_guard<std::mutex> guard{plottingMutex};
        cudaMemcpy(const_cast<float *>(x.data()), devX, N * sizeof(float),
                   cudaMemcpyDeviceToHost);
        CHECK_ERR;
        status = PlottingStatus::UPDATED;
    }

    // clang-format off
    cudaFree(devX); CHECK_ERR;
    cudaFree(devY); CHECK_ERR;
    cudaFree(devMatrix.indices); CHECK_ERR;
    cudaFree(devMatrix.values); CHECK_ERR;
    CUBLAS_CHECK(cublasDestroy(cublasHandle));
    // clang-format on
}
} // namespace ht