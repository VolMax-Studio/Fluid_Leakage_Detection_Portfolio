#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include "DwtCusumEngine.h"

// Struct to store CSV row data
struct RowData {
    double timestamp_h;
    double raw_pressure;
    double predicted_baseline;
    double sigma;
};

int main(int argc, char* argv[]) {
    std::string filename = "edge/sensor_test_data.csv";
    if (argc > 1) {
        filename = argv[1];
    }

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return 1;
    }

    std::cout << "========================================================\n";
    std::cout << "  C++ Standalone Edge DWT + CUSUM Leak Detection Engine \n";
    std::cout << "========================================================\n";
    std::cout << "Replaying pressure trace from: " << filename << "\n\n";

    // Read CSV header
    std::string line;
    std::getline(file, line);

    // Instantiate the engine
    // cusum_drift = 0.5, cusum_threshold = 8.0, dwt_threshold = 0.5
    DwtCusumEngine engine(0.5, 8.0, 0.5);

    double first_leak_alarm_h = -1.0;
    double first_burst_alarm_h = -1.0;
    int leak_alarm_count = 0;
    int burst_alarm_count = 0;
    bool prev_leak_state = false;

    int total_steps = 0;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        
        RowData row;
        try {
            std::getline(ss, cell, ',');
            row.timestamp_h = std::stod(cell);
            std::getline(ss, cell, ',');
            row.raw_pressure = std::stod(cell);
            std::getline(ss, cell, ',');
            row.predicted_baseline = std::stod(cell);
            std::getline(ss, cell, ',');
            row.sigma = std::stod(cell);
        } catch (...) {
            continue; // Skip malformed rows
        }

        // Feed to edge engine
        engine.update(row.raw_pressure, row.predicted_baseline, row.sigma);
        total_steps++;

        // Detect transitions
        bool current_leak = engine.hasLeakAlarm();
        if (current_leak && !prev_leak_state) {
            leak_alarm_count++;
            if (first_leak_alarm_h < 0.0) {
                first_leak_alarm_h = row.timestamp_h;
            }
            std::cout << "[ALARM: LEAK] CUSUM accumulated sum reached threshold (" 
                      << engine.getCusumValue() << ") at t = " << row.timestamp_h 
                      << " h (Pressure drop residual: " << engine.getResidual() << " m)\n";
        }
        prev_leak_state = current_leak;

        if (engine.hasBurstAlarm()) {
            burst_alarm_count++;
            if (first_burst_alarm_h < 0.0) {
                first_burst_alarm_h = row.timestamp_h;
            }
            std::cout << "[ALARM: BURST] DWT detail magnitude (" << std::abs(engine.getDwtDetail())
                      << ") exceeded threshold at t = " << row.timestamp_h << " h\n";
        }
    }

    file.close();

    std::cout << "\n--------------------------------------------------------\n";
    std::cout << "  REPLAY COMPLETED (" << total_steps << " steps)\n";
    std::cout << "--------------------------------------------------------\n";
    std::cout << "  CUSUM Leak Alarms Fired:  " << leak_alarm_count << "\n";
    if (first_leak_alarm_h >= 0.0) {
        std::cout << "  First Leak Alarm Time:    " << first_leak_alarm_h << " h\n";
    }
    std::cout << "  DWT Burst Alarms Fired:   " << burst_alarm_count << "\n";
    if (first_burst_alarm_h >= 0.0) {
        std::cout << "  First Burst Alarm Time:   " << first_burst_alarm_h << " h\n";
    }
    std::cout << "========================================================\n";

    return 0;
}
