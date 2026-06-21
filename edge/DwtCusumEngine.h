#ifndef DWT_CUSUM_ENGINE_H
#define DWT_CUSUM_ENGINE_H

class DwtCusumEngine {
public:
    DwtCusumEngine(double cusum_drift = 0.5, 
                   double cusum_threshold = 8.0, 
                   double dwt_threshold = 0.5);

    /**
     * Feed a raw pressure sample and baseline info into the engine.
     * 
     * @param pressure Raw pressure reading [m H2O]
     * @param baseline Estimated baseline pressure [m H2O]
     * @param std_dev Standard deviation of normal pressure fluctuations [m H2O]
     */
    void update(double pressure, double baseline, double std_dev);

    /**
     * Reset the engine state.
     */
    void reset();

    // Getters for states and outputs
    bool hasLeakAlarm() const { return leak_alarm_; }
    bool hasBurstAlarm() const { return burst_alarm_; }
    double getCusumValue() const { return S_; }
    double getDwtDetail() const { return dwt_detail_; }
    double getResidual() const { return residual_; }

private:
    // Parameters
    double k_; // CUSUM drift/allowance
    double H_; // CUSUM alarm threshold
    double dwt_threshold_; // Haar DWT level-1 detail alarm threshold

    // States
    double S_; // CUSUM cumulative sum
    double prev_pressure_;
    bool has_prev_pressure_;
    int sample_count_;
    double dwt_detail_;
    double residual_;

    bool leak_alarm_;
    bool burst_alarm_;
};

#endif // DWT_CUSUM_ENGINE_H
