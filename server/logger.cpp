#include "logger.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <sys/stat.h>

Logger& Logger::instance() {
    static Logger instance;
    return instance;
}

void Logger::init() {
    mkdir("logs", 0777);
    std::lock_guard<std::mutex> lock(logMutex);
    rotateLogFile(); // Initial creation
}

void Logger::shutdown() {
    std::lock_guard<std::mutex> lock(logMutex);
    if (globalLogFile.is_open()) {
        flushStatsAndReset();
        globalLogFile.close();
    }
}

void Logger::write(const std::string& message) {
    std::lock_guard<std::mutex> lock(logMutex); 
    if (globalLogFile.is_open()) {
        globalLogFile << message << std::endl;
        globalLogFile.flush(); 
    }
}

void Logger::write(const std::string& message, const std::string& channelKey) {
    std::lock_guard<std::mutex> lock(logMutex);
    if (globalLogFile.is_open()) {
        globalLogFile << message << std::endl;
        globalLogFile.flush();
    }

    auto& channelFile = getChannelLog(channelKey);
    if (channelFile.is_open()) {
        channelFile << message << std::endl;
        channelFile.flush();
    }
}

void Logger::recordKernelStat(const std::string& kernelType) {
    std::lock_guard<std::mutex> lock(statsMutex);
    currentLogKernelStats[kernelType]++;
}

void Logger::recordConnectionStat(const std::string& clientKey) {
    std::lock_guard<std::mutex> lock(statsMutex);
    currentLogConnectionStats[clientKey]++;
}

long long Logger::incrementConnectionCount() {
    return connectionCount.fetch_add(1, std::memory_order_relaxed);
}

// 内部辅助函数：必须在持有 logMutex 下调用
void Logger::flushStatsAndReset() {
    std::lock_guard<std::mutex> lock(statsMutex); // 锁住统计数据

    if (!globalLogFile.is_open()) return;

    globalLogFile << "\n-------------------------------------------------------\n";
    globalLogFile << "      Session Statistics\n";
    globalLogFile << "-------------------------------------------------------\n";
    globalLogFile << "Total Connections: " << connectionCount.load() << "\n";

    if (!currentLogConnectionStats.empty()) {
        globalLogFile << "\nConnections by Client:\n";
        for (const auto& item : currentLogConnectionStats) {
            globalLogFile << "  " << item.first << ": " << item.second << " session(s)\n";
        }
    }

    globalLogFile << "\n-------------------------------------------------------\n";
    globalLogFile << "      Kernel Statistics\n";
    globalLogFile << "-------------------------------------------------------\n";

    if (currentLogKernelStats.empty()) {
        globalLogFile << "No kernels recorded in this session.\n";
    } else {
        using PairType = std::pair<std::string, long long>;
        std::vector<PairType> sortedStats(currentLogKernelStats.begin(), currentLogKernelStats.end());

        std::sort(sortedStats.begin(), sortedStats.end(), 
            [](const PairType& a, const PairType& b) {
                return a.second > b.second;
            });

        globalLogFile << std::left << std::setw(50) << "Kernel Name" << " | " << "Count" << "\n";
        globalLogFile << "----------------------------------------------|--------\n";
        
        long long total = 0;
        for (const auto& item : sortedStats) {
            globalLogFile << std::left << std::setw(45) << item.first << " | " << item.second << "\n";
            total += item.second;
        }
        globalLogFile << "----------------------------------------------|--------\n";
        globalLogFile << std::left << std::setw(45) << "TOTAL" << " | " << total << "\n";
    }
    
    globalLogFile << "-------------------------------------------------------\n\n";
    globalLogFile.flush();

    currentLogKernelStats.clear();
    currentLogConnectionStats.clear();
}

std::string Logger::getCurrentTimeStrForFile() {
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time);
    std::stringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
    return ss.str();
}

void Logger::rotateLogFile() {
    // 调用者已持有 logMutex
    if (globalLogFile.is_open()) {
        flushStatsAndReset();
        globalLogFile.close();
        std::cout << "[Logger] Rotated log file." << std::endl;
    }

    currentLogSuffix = getCurrentTimeStrForFile();
    closeChannelLogs();

    std::string filename = "logs/" + currentLogSuffix + ".log";
    globalLogFile.open(filename, std::ios::out | std::ios::app);
    
    if (!globalLogFile.is_open()) {
        std::cerr << "[Logger] Fatal: Cannot create " << filename << std::endl;
    } else {
        std::cout << "[Logger] New log file: " << filename << std::endl;
    }
}

std::string Logger::sanitizeKey(const std::string& key) {
    std::string sanitized = key;
    for (auto& ch : sanitized) {
        if (ch == '/' || ch == '\\' || ch == ' ') ch = '_';
    }
    if (sanitized.empty()) sanitized = "unknown";
    return sanitized;
}

std::ofstream& Logger::getChannelLog(const std::string& channelKey) {
    std::string safeKey = sanitizeKey(channelKey);
    auto it = channelLogFiles.find(safeKey);
    if (it == channelLogFiles.end() || !it->second.is_open()) {
        // 若尚未初始化时间戳，确保有后缀
        if (currentLogSuffix.empty()) currentLogSuffix = getCurrentTimeStrForFile();
        std::string filename = "logs/" + currentLogSuffix + "_" + safeKey + ".log";
        std::ofstream ofs(filename, std::ios::out | std::ios::app);
        if (!ofs.is_open()) {
            std::cerr << "[Logger] Fatal: Cannot create " << filename << std::endl;
            channelLogFiles[safeKey] = std::ofstream(); // 占位，避免反复尝试
        } else {
            std::cout << "[Logger] New channel log file: " << filename << std::endl;
            channelLogFiles[safeKey] = std::move(ofs);
        }
        it = channelLogFiles.find(safeKey);
    }
    return it->second;
}

void Logger::closeChannelLogs() {
    for (auto& item : channelLogFiles) {
        if (item.second.is_open()) item.second.close();
    }
    channelLogFiles.clear();
}

Logger::~Logger() {
    shutdown();
}