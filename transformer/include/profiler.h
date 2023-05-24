#include <chrono>
#include <iostream>
#include <map>
#include <string>

class Profiler {
   public:
    static Profiler& getInstance() {
        static Profiler instance;
        return instance;
    }

    void start(const std::string& section) { start_times[section] = std::chrono::high_resolution_clock::now(); }

    void start(const std::string& section, const long long section_flops) {
        start_times[section] = std::chrono::high_resolution_clock::now();
        if (flops.count(section) == 0)
            flops[section] = section_flops;
        else
            flops[section] += section_flops;
    }

    void reset() {
        start_times.clear();
        durations.clear();
        counts.clear();
        flops.clear();
    }

    void stop(const std::string& section) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_times[section]).count();
        durations[section] += duration;
        counts[section]++;
    }

    void report_internal() const {
        std::cout << "Section, Total time(us), Average time(us), Count, GOPs" << std::endl;
        for (const auto& entry : durations) {
            std::string row;
            row += entry.first + ", ";
            row += std::to_string(entry.second) + ", ";
            row += std::to_string(entry.second / counts.at(entry.first)) + ", ";
            if (flops.count(entry.first) == 0)
                row += std::to_string(counts.at(entry.first)) + ", N/A, N/A";
            else {
                row += std::to_string(counts.at(entry.first)) + ", ";
                // ops and microsecond
                row += std::to_string((((float)flops.at(entry.first)) / (float)(entry.second)) / 1000.0);
            }
            std::cout << row << std::endl;
        }
    }

    void report() const {
#ifdef PROFILER
        report_internal();
#endif
    }

   private:
    Profiler() {}
    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;

    std::map<std::string, std::chrono::high_resolution_clock::time_point> start_times;
    std::map<std::string, long long> flops;
    std::map<std::string, long long> durations;
    std::map<std::string, int> counts;
};
