#include "aux.h"
#include "gtest/gtest.h"

TEST(Reduction, Power2) {
    EXPECT_EQ(getNearestPow2Number(1025), 2048);
    EXPECT_EQ(getNearestPow2Number(1000), 1024);
    EXPECT_EQ(getNearestPow2Number(513), 1024);
    EXPECT_EQ(getNearestPow2Number(512), 512);
    EXPECT_EQ(getNearestPow2Number(511), 512);
}