#ifndef SPARSE_TEST_AUX_H
#define SPARSE_TEST_AUX_H

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <vector>

using namespace ::testing;

std::vector<Matcher<float>> getExpectedArray(const std::vector<float> &expectedVector);

#endif // SPARSE_TEST_AUX_H
