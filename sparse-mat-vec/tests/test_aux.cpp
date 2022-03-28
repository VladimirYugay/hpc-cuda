#include "test_aux.h"

std::vector<Matcher<float>> getExpectedArray(const std::vector<float> &expectedVector) {
    std::vector<Matcher<float>> expectedArray;
    for (const auto item : expectedVector) {
        expectedArray.emplace_back(item);
    }
    return expectedArray;
}
