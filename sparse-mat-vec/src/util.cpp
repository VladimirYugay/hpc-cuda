#include "util.h"
#include "aux.h"
#include "boost/filesystem.hpp"
#include "external/mmio.h"
#include <boost/program_options.hpp>
#include <cuda.h>
#include <map>
#include <utility>

namespace po = boost::program_options;

std::string modeToStr(ExecutionMode mode) {
    static std::map<ExecutionMode, std::string> map{
        {ExecutionMode::PAGERANK, "pagerank"},
        {ExecutionMode::PAGERANK, "pagerank-vectorized"},
        {ExecutionMode::HEAT_BAND, "heat-band"},
        {ExecutionMode::HEAT_CUSPARSE, "heat-cusparse"},
        {ExecutionMode::HEAT_ELLPACK, "heat-ellpack"}};
    return map[mode];
}

Configuration makeConfig(int argc, char **argv) {
    Configuration config;

    po::options_description description("HPC-AA Sparse Linear Algebra Exercise.");

    std::string executionMode{};
    long int numGridPoints{};
    long int stepsPerPrint{};
    long int numIterations{};
    // clang-format off
    description.add_options()
        ("help,h", "produce help message")
        ("mode,m", po::value<std::string>(&executionMode)->default_value("pagerank"), "Execution modes: pagerank, pagerank-vectorized, heat-band, heat-cusparse, heat-ellpack.")
        ("pagerank-matrix,f", po::value<std::string>(&config.matrixFile)->default_value(""), "Specify path to the Matrix the PageRank algorithm is to be executed on. Must be in MatrixMarket format.")
        ("iterations,i", po::value<long int>(&numIterations)->default_value(-1), "Num. iterations to run. Stop is based on eps. and err. if a negative value provided.")
        ("heat-grid-points,n", po::value<long int>(&numGridPoints)->default_value(0), "Num. grid points for the heat equation calculation.")
        ("print-rate,p", po::value<long int>(&stepsPerPrint)->default_value(500), "Number of steps between printing results (-1 to switch off)")
        ;
    // clang-format on

    po::variables_map cmdMap;
    po::store(po::parse_command_line(argc, argv, description), cmdMap);
    po::notify(cmdMap);

    if (cmdMap.count("help")) {
        std::cout << description << std::endl;
        exit(EXIT_SUCCESS);
    }

    std::map<std::string, ExecutionMode> table{{"pagerank", ExecutionMode::PAGERANK},
                                               {"pagerank-vectorized", ExecutionMode::PAGERANK},
                                               {"heat-band", ExecutionMode::HEAT_BAND},
                                               {"heat-cusparse", ExecutionMode::HEAT_CUSPARSE},
                                               {"heat-ellpack", ExecutionMode::HEAT_ELLPACK}};

    if (table.find(executionMode) != table.end()) {
        config.executionMode = table[executionMode];
    } else {
        std::stringstream stream;
        stream << "unknown execution mode provided: " << executionMode << ". Allowed: ";
        for (const auto &item : table) {
            stream << item.first << ", ";
        }
        throw std::runtime_error(stream.str());
    }

    if (numGridPoints < 0) {
        throw std::runtime_error("Provided a negative number of grid points");
    } else {
        config.numGridPoints = numGridPoints;
    }

    if (numIterations > 0) {
        config.numIterations = numIterations;
    }

    if (stepsPerPrint < 0) {
        std::cout << "Info: provided a negative value for plot rate. Printing will be turned off\n";
        config.stepsPerPrint = std::numeric_limits<size_t>::max();
    } else {
        config.stepsPerPrint = stepsPerPrint;
    }

    return config;
}

void checkConfig(const Configuration &config) {
    std::stringstream stream;
    if (!config.matrixFile.empty() && (config.numGridPoints > 0)) {
        stream << "Must either provide a matrix for PageRank, "
               << "or a grid Size for the Heat Equation Solver. You provided both";
        throw std::runtime_error(stream.str());
    }

    if (config.matrixFile.empty() && (config.numGridPoints == 0)) {
        stream << "Must either provide a matrix for PageRank, "
               << "or a grid Size for the Heat Equation Solver. You provided None.";
    }

    if ((config.executionMode == ExecutionMode::PAGERANK) ||
        (config.executionMode == ExecutionMode::PAGERANK_VECTORIZED)) {
        if (config.matrixFile.empty()) {
            throw std::runtime_error("path to a matrix is not provided for PageRank mode");
        }
    } else {
        if (config.numGridPoints == 0) {
            throw std::runtime_error("num. grid point is not provided for Heat Equation solver");
        }
    }
}


std::ostream &operator<<(std::ostream &stream, const Configuration &config) {
    stream << "execution mode: " << modeToStr(config.executionMode) << '\n'
           << "matrix file: " << config.matrixFile << '\n'
           << "num. grid points: " << config.numGridPoints;
    return stream;
}


CsrMatrix loadMarketMatrix(const std::string &fileName) {
    CsrMatrix matrix;

    try {
        std::stringstream stream;

        if (!boost::filesystem::exists(fileName)) {
            stream << "File \'" << fileName << "\' doesn't exist";
            throw std::runtime_error(stream.str());
        }

        // NOTE: the file should have already been check
        FILE *file = fopen(fileName.c_str(), "r");
        if (file == NULL) {
            stream << "cannot open file: " << fileName << " to read. Execution is aborted";
            throw std::runtime_error(stream.str());
        }

        MM_typecode matcode;
        if (mm_read_banner(file, &matcode) != 0) {
            throw std::runtime_error("Could not process Matrix Market banner");
        }

        /*  This is how one can screen matrix types if their application */
        /*  only supports a subset of the Matrix Market data types.      */
        if (mm_is_complex(matcode) && mm_is_matrix(matcode) && mm_is_sparse(matcode)) {
            stream << "Sorry, this application does not support "
                   << "Market Market type: " << mm_typecode_to_str(matcode);
            throw std::runtime_error(stream.str());
        }

        /* find out size of sparse matrix .... */
        int M{}, N{}, nnz{};
        int returnCode = mm_read_mtx_crd_size(file, &M, &N, &nnz);
        if (returnCode != 0) {
            stream << "failed to read matrix. Error" << returnCode;
            throw std::runtime_error(stream.str());
        }

        if (M != N) {
            stream << "Error: matrix is not square. Given, M = " << M << ", N = " << N;
            throw std::runtime_error(stream.str());
        }


        std::vector<int> I{};
        std::vector<int> J{};
        std::vector<float> values{};
        std::vector<int> ptr(N + 1, 0);
        if (mm_is_symmetric(matcode)) {
            nnz *= 2;

            /* reserve memory for host matrices */
            I.resize(nnz, 0);
            J.resize(nnz, 0);
            values.resize(nnz, 0.0f);

            // read non-zero matrix entries
            for (int i = 0; i < nnz; i += 2) {
                // transpose matrix while reading
                fscanf(file, "%d %d\n", &J[i], &I[i]);
                --I[i]; /* adjust from 1-based to 0-based */
                --J[i];
                I[i + 1] = J[i];
                J[i + 1] = I[i];
                values[i] = 1;
                values[i + 1] = 1;
            }
        } else {

            /* reserve memory for host matrices */
            I.resize(nnz, 0);
            J.resize(nnz, 0);
            values.resize(nnz, 0.0f);

            // read non-zero matrix entries
            for (int i = 0; i < nnz; ++i) {
                // transpose matrix while reading
                fscanf(file, "%d %d\n", &J[i], &I[i]);
                --I[i]; /* adjust from 1-based to 0-based */
                --J[i];
                values[i] = 1;
            }
        }
        fclose(file);

        // write out matrix information
        mm_write_banner(stdout, matcode);
        mm_write_mtx_crd_size(stdout, N, N, nnz);

        matrix.numRows = N;
        matrix.nnz = nnz;
        matrix.indices = std::move(J);
        matrix.start = std::move(ptr);
        matrix.values = std::move(values);

        convertCooToCsr(matrix, I);
    } catch (std::exception &error) {
        std::stringstream stream;
        stream << "Execution aborted while reading a matrix from a file: " << error.what();
        throw std::runtime_error(stream.str());
    }

    return matrix;
}
