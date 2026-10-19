#pragma once 

#include <vector>
#include <cstdint>
#include <string>

namespace risk { 

struct AssetData {
    double spot; 
    double weight;
    double vol;
    double drift;
};

class SimulationEngine {
private:
    int num_threads_;
    uint64_t seed_;

public:
    explicit SimulationEngine(int num_threads = 0, uint64_t seed = 42);
    std::vector<double> 
    simulate_portfolio_values( const std::vector<AssetData>&assets,
    const std::vector<double>&correlation_matrix, int num_paths, 
    double horizon_days =1.0, bool use_antithetic = true);
};
}