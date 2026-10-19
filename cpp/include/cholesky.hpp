#pragma once

#include <vector>

namespace risk {

class Cholesky {
public:
    // Cholesky factorization: Sigma = L * L^T for flat 1D n x n matrix
    // Adds diagonal jitter to guarantee positive-definiteness on empirical covariance
    static std::vector<double> decompose(const std::vector<double>& matrix, int n, double jitter = 1e-10);

    // Correlates independent standard normal shocks: z_out = L * z_in
    // Exploits lower-triangular structure (loop bounds j <= i)
    static void correlate(const std::vector<double>& L, const double* z_in, double* z_out, int n);
};

} // namespace risk
