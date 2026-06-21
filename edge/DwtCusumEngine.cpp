#include "DwtCusumEngine.h"
#include <cmath>

DwtCusumEngine::DwtCusumEngine(double cusum_drift, double cusum_threshold, double dwt_threshold)
    : k_(cusum_drift),
      H_(cusum_threshold),
      dwt_threshold_(dwt_threshold),
      S_(0.0),
      prev_pressure_(0.0),
      has_prev_pressure_(false),
      sample_count_(0),
      dwt_detail_(0.0),
      residual_(0.0),
      leak_alarm_(false),
      burst_alarm_(false) {}

void DwtCusumEngine::reset() {
    S_ = 0.0;
    prev_pressure_ = 0.0;
    has_prev_pressure_ = false;
    sample_count_ = 0;
    dwt_detail_ = 0.0;
    residual_ = 0.0;
    leak_alarm_ = false;
    burst_alarm_ = false;
}

void DwtCusumEngine::update(double pressure, double baseline, double std_dev) {
    // 1. CUSUM Leak Detection
    residual_ = pressure - baseline;
    double std_val = (std_dev > 1e-6) ? std_dev : 1.0;
    double z = residual_ / std_val;

    // Accumulate negative shift
    // S_t = max(0, S_{t-1} - z_t - k)
    double S_next = S_ - z - k_;
    if (S_next < 0.0) {
        S_next = 0.0;
    }
    S_ = S_next;

    if (S_ > H_) {
        leak_alarm_ = true;
    } else {
        leak_alarm_ = false;
    }

    // 2. Haar DWT Level-1 Detail Burst Detection
    burst_alarm_ = false;
    if (has_prev_pressure_) {
        // We compute DWT coefficients every 2 samples (decimated)
        if (sample_count_ % 2 == 1) {
            // Haar level-1 detail: (x[2k] - x[2k+1]) / sqrt(2.0)
            dwt_detail_ = (prev_pressure_ - pressure) / 1.4142135623730951;
            
            if (std::abs(dwt_detail_) > dwt_threshold_) {
                burst_alarm_ = true;
            }
        }
    }

    prev_pressure_ = pressure;
    has_prev_pressure_ = true;
    sample_count_++;
}
