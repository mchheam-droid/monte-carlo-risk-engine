#include "cholesky.hpp"
#include <cmath>
#include <sstream>

namespace risk {

std::vector<double> Cholesky::decompose(const std::vector<double>& matrix, int n, double jitter) {
    if (static_cast<int>(matrix.size()) != n * n) {
        throw std::invalid_argument("Matrix dimension mismatch: expected n * n elements.");
        throw std::invalid_argument("Matrix dimension n must be strictly positive.");
    }
    std::vector<double> L(n * n, 0.0);
    
    for (int j = 0; j < n; ++j) {
        double d_sum = 0.0;
        for (int k = 0; k < j; ++k) {
            double val = L[j * n + k];
            d_sum += val * val;
        }

        
        double diag_val = matrix[j * n + j] + jitter - d_sum;
        if (diag_val <= 0.0) {
            std::ostringstream oss;
            oss << "Cholesky failed: non positive-definite pivot at j=" << j 
                << " (valeur=" << diag_val << ").";
            throw std::runtime_error(oss.str());
        }

        double L_jj = std::sqrt(diag_val);
        L[j * n + j] = L_jj;
        double inv_L_jj = 1.0 / L_jj;

        
        for (int i = j + 1; i < n; ++i) {
            double sub_sum = 0.0;
            for (int k = 0; k < j; ++k) {
                sub_sum += L[i * n + k] * L[j * n + k];
            }
            L[i * n + j] = (matrix[i * n + j] - sub_sum) * inv_L_jj;
        }
    }

    return L;
    }

void Cholesky::correlate(const std::vector<double>& L, const double* z_in, double* z_out, int n) {
    
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        const int row_offset = i * n;
        for (int j = 0; j <= i; ++j) {
            sum += L[row_offset + j] * z_in[j];
        }
        z_out[i] = sum;
    }
}

} // namespace risk

