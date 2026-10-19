#include "simulation_engine.hpp"
#include "cholesky.hpp"
#include "rng.hpp"

#include <thread>
#include <cmath>
#include <algorithm>

namespace risk {

SimulationEngine::SimulationEngine(int num_threads, uint64_t seed) 
    : seed_(seed) {
    if (num_threads <= 0) {
        // Automatically detect available CPU hardware threads (e.g. 16 cores)
        num_threads_ = std::max(1u, std::thread::hardware_concurrency());
    } else {
        num_threads_ = num_threads;
    }
}

std::vector<double> SimulationEngine::simulate_portfolio_values(
    const std::vector<AssetData>& assets,
    const std::vector<double>& correlation_matrix,
    int num_paths,
    double horizon_days,
    bool use_antithetic
) {
    const int n_assets = static_cast<int>(assets.size());
    const double dt = horizon_days / 252.0; // 252 business days per calendar year
    const double sqrt_dt = std::sqrt(dt);

    // 1. Cholesky factorization of the correlation matrix (computed only once)
    std::vector<double> L = Cholesky::decompose(correlation_matrix, n_assets);

    // 2. Precompute drift and volatility constants to maximize inner loop throughput
    std::vector<double> drift_terms(n_assets);
    std::vector<double> vol_terms(n_assets);
    for (int i = 0; i < n_assets; ++i) {
        drift_terms[i] = (assets[i].drift - 0.5 * assets[i].vol * assets[i].vol) * dt;
        vol_terms[i] = assets[i].vol * sqrt_dt;
    }

    // Output buffer: contiguous memory for all simulated terminal portfolio values
    std::vector<double> portfolio_values(num_paths, 0.0);

    // 3. Partition workload evenly across threads
    const int actual_threads = std::min(num_threads_, num_paths);
    const int paths_per_thread = num_paths / actual_threads;

    std::vector<std::thread> workers;
    workers.reserve(actual_threads);

    for (int t = 0; t < actual_threads; ++t) {
        int start_idx = t * paths_per_thread;
        int end_idx = (t == actual_threads - 1) ? num_paths : start_idx + paths_per_thread;
        uint64_t thread_seed = seed_ + t * 10007; // Unique seed per worker to guarantee independent streams

        workers.emplace_back([&, start_idx, end_idx, thread_seed]() {
            ThreadRNG local_rng(thread_seed);

            std::vector<double> z_indep(n_assets);
            std::vector<double> z_corr(n_assets);
            std::vector<double> z_indep_anti(n_assets);
            std::vector<double> z_corr_anti(n_assets);

            int p = start_idx;
            while (p < end_idx) {
                if (use_antithetic && (p + 1 < end_idx)) {
                    // Antithetic variates: draw paired shocks (+Z, -Z) to reduce Monte Carlo variance
                    local_rng.fill_antithetic_pair(z_indep.data(), z_indep_anti.data(), n_assets);

                    // Path 1 (+Z)
                    Cholesky::correlate(L, z_indep.data(), z_corr.data(), n_assets);
                    double val1 = 0.0;
                    for (int i = 0; i < n_assets; ++i) {
                        double st = assets[i].spot * std::exp(drift_terms[i] + vol_terms[i] * z_corr[i]);
                        val1 += assets[i].weight * st;
                    }
                    portfolio_values[p] = val1;

                    // Path 2 (-Z)
                    Cholesky::correlate(L, z_indep_anti.data(), z_corr_anti.data(), n_assets);
                    double val2 = 0.0;
                    for (int i = 0; i < n_assets; ++i) {
                        double st = assets[i].spot * std::exp(drift_terms[i] + vol_terms[i] * z_corr_anti[i]);
                        val2 += assets[i].weight * st;
                    }
                    portfolio_values[p + 1] = val2;

                    p += 2;
                } else {
                    // Standard simulation path
                    local_rng.fill_normals(z_indep.data(), n_assets);
                    Cholesky::correlate(L, z_indep.data(), z_corr.data(), n_assets);

                    double val = 0.0;
                    for (int i = 0; i < n_assets; ++i) {
                        double st = assets[i].spot * std::exp(drift_terms[i] + vol_terms[i] * z_corr[i]);
                        val += assets[i].weight * st;
                    }
                    portfolio_values[p] = val;
                    p += 1;
                }
            }
        });
    }

    // 4. Synchronize all worker threads
    for (auto& w : workers) {
        if (w.joinable()) {
            w.join();
        }
    }

    return portfolio_values;
}

} // namespace risk

