#include "solver.h"
#include <cmath>

using namespace std::chrono;

double Solver::runCpu() {

    std::vector<float> swapDataField = dataField;

    auto swap = reinterpret_cast<float(*)[settings.num1dGridPoints]>(&swapDataField[0]);
    auto field = reinterpret_cast<float(*)[settings.num1dGridPoints]>(&dataField[0]);

    float err = std::numeric_limits<float>::max();
    float simulation_time{};
    high_resolution_clock::time_point start = high_resolution_clock::now();
    for (size_t counter{}; ((err > settings.eps) && (settings.numIterations > counter) &&
                            (settings.endTime > simulation_time));
         ++counter) {

        err = 0.0f;
        for (int j = 1; j < (settings.num1dGridPoints - 1); ++j) {
            for (int i = 1; i < (settings.num1dGridPoints - 1); ++i) {

                float align =
                    4.0f * (field[j][i - 1] + field[j][i + 1] + field[j - 1][i] + field[j + 1][i]);

                float cross = field[j - 1][i - 1] + field[j - 1][i + 1] + field[j + 1][i - 1] +
                              field[j + 1][i + 1];

                swap[j][i] = field[j][i] + params.factor * (align + cross - 20.0f * field[j][i]) *
                                               params.invDhSquare / 6.0f;
                err = std::max(err, static_cast<float>(fabs(field[j][i] - swap[j][i])));
            }
        }

        auto tmp = reinterpret_cast<float *>(swap);
        swap = reinterpret_cast<decltype(swap)>(field);
        field = reinterpret_cast<decltype(field)>(tmp);

        if ((counter % settings.stepsPerPrint) == 0) {
            {
                std::lock_guard<std::mutex> guard(plottingMutex);
                solutionField = swapDataField;
            }

            if (settings.isVerbose) {
                std::cout << "iteration: " << counter << "; simulation time, s: " << simulation_time
                          << "; max difference = " << err << std::endl;
            }
        }
        simulation_time += params.dtStable;
    }
    high_resolution_clock::time_point end = high_resolution_clock::now();

    if (settings.isVerbose) {
        std::cout << "done...\n";
    }
    return duration_cast<duration<double>>(end - start).count();
}
