#include "kernels/kernels.h"
#include "solver.h"
using namespace std::chrono;

double Solver::runGpu(ExecutionMode mode) {

    float *devDataField{nullptr};
    cudaMalloc(&devDataField, dataField.size() * sizeof(float));
    CHECK_ERR;
    cudaMemcpy(devDataField, dataField.data(), dataField.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    CHECK_ERR;

    float *devSwapField{nullptr};
    cudaMalloc(&devSwapField, dataField.size() * sizeof(float));
    CHECK_ERR;
    cudaMemcpy(devSwapField, devDataField, dataField.size() * sizeof(float),
               cudaMemcpyDeviceToDevice);
    CHECK_ERR;

    float *errField{nullptr};
    cudaMalloc(&errField, dataField.size() * sizeof(float));
    CHECK_ERR;
    Reducer reducer(dataField.size());

    float err = std::numeric_limits<float>::max();
    float simulation_time{};

    if (settings.isVerbose) {
        std::cout << "solver starts...\n";
    }

    high_resolution_clock::time_point start = high_resolution_clock::now();
    for (size_t counter{}; ((err > settings.eps) && (settings.numIterations > counter) &&
                            (settings.endTime > simulation_time));
         ++counter) {

        // compute a filed
        switch (mode) {
            case ExecutionMode::GPU_GLOB_MEM: {
                launch_gpuSimple(devSwapField, devDataField, settings, params);
                break;
            }
            case ExecutionMode::GPU_SHR_MEM: {
                launch_gpuWithShrMem(devSwapField, devDataField, settings, params);
                break;
            }
        }

        std::swap(devDataField, devSwapField);
        if ((counter % settings.stepsPerPrint) == 0) {
            {
                std::lock_guard<std::mutex> guard(plottingMutex);
                cudaMemcpy(const_cast<float *>(solutionField.data()), devSwapField,
                           solutionField.size() * sizeof(float), cudaMemcpyDeviceToHost);
                CHECK_ERR;
            }

            // compute max error
            launch_computeErr(errField, devSwapField, devDataField, dataField.size());
            err = reducer.reduceMax(errField);

            if (settings.isVerbose) {
                std::cout << "iteration: " << counter << "; simulation time, s: " << simulation_time
                          << "; max difference = " << err << std::endl;
            }
        }
        simulation_time += params.dtStable;
    }

    {
        std::lock_guard<std::mutex> guard(plottingMutex);
        cudaMemcpy(const_cast<float *>(solutionField.data()), devDataField,
                   solutionField.size() * sizeof(float), cudaMemcpyDeviceToHost);
        CHECK_ERR;
    }

    high_resolution_clock::time_point end = high_resolution_clock::now();
    if (settings.isVerbose) {
        std::cout << "done...\n";
    }

    cudaFree(devDataField);
    CHECK_ERR;
    cudaFree(devSwapField);
    CHECK_ERR;

    return duration_cast<duration<double>>(end - start).count();
}