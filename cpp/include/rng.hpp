#pragma once 
#include <random>
#include <vector>
#include <cstdint>

namespace risk {

class ThreadRNG {
private :
    std ::mt19937_64 rng;
    std::normal_distribution<double>
normal_dist_;

public:
    explicit ThreadRNG(uint64_t seed = 42)
    : rng(seed), normal_dist_(0.0, 1.0) {}
    double next_normal() {
        return normal_dist_(rng);
    }   
    void fill_normals(double* buffer, int n ){
        for (int i = 0; i<n; ++i){
            buffer[i] = normal_dist_(rng);

        }
    }
    void fill_antithetic_pair(double*pos,double*neg, int n){
        for (int i = 0; i<n; ++i){
            double z = normal_dist_(rng);
            pos[i] = z;
            neg[i] = -z;
        }
    }
    
};
}