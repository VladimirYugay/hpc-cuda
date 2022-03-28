#include "data_types.h"
#include "kernels/kernels.h"
#include "solvers/heat_transfer/ht.h"
#include "solvers/pagerank/pr.h"
#include "util.h"
#include <iostream>

int main(int argc, char *argv[]) {
    try {
        Configuration config = makeConfig(argc, argv);
        checkConfig(config);

        std::cout << "================= SPARSE LINEAR ALGEBRA CONFIGURATION ==================\n";
        std::cout << config << '\n';
        std::cout << "device: " << getDeviceName() << '\n';
        std::cout << "========================================================================"
                  << std::endl;

        double time{};
        if ((config.executionMode == ExecutionMode::PAGERANK) ||
            (config.executionMode == ExecutionMode::PAGERANK_VECTORIZED)) {
            CsrMatrix matrix = loadMarketMatrix(config.matrixFile);

            pr::Settings settings;
            settings.numIterations = config.numIterations;

            time = pr::pageRank(matrix, config.executionMode, settings);
        } else {
            ht::Settings settings{};
            settings.numIterations = config.numIterations;
            settings.stepsToPrint = config.stepsPerPrint;

            ht::PoissonSolver solver(settings);
            time = solver.run(config.numGridPoints, config.executionMode);
        }
        std::cout << "Total time: " << time << ", s" << std::endl;

    } catch (std::exception &error) {
        std::cout << "Error: " << error.what() << std::endl;
        exit(EXIT_FAILURE);
    }
    return 0;
}