#pragma once

#include <string>
#include <mutex>
#include <fstream>
#include <map>
#include <atomic>
#include <vector>

class Logger {
public:
    static Logger& instance();

    void init();
    void write(const std::string& message);
    // 按通道写日志（同时仍写入全局日志），channelKey 通常用 shm 名称
    void write(const std::string& message, const std::string& channelKey);
    void rotateLogFile();
    void recordKernelStat(const std::string& kernelType);
    void recordConnectionStat(const std::string& clientKey);
    long long incrementConnectionCount();
    
    // 关闭时调用
    void shutdown();

private:
    Logger() = default;
    ~Logger();
    
    void flushStatsAndReset();
    std::string getCurrentTimeStrForFile();
    std::string sanitizeKey(const std::string& key);
    std::ofstream& getChannelLog(const std::string& channelKey);
    void closeChannelLogs();

    std::mutex logMutex;
    std::ofstream globalLogFile;
    std::map<std::string, std::ofstream> channelLogFiles;
    std::string currentLogSuffix;
    
    std::mutex statsMutex;
    std::map<std::string, long long> currentLogKernelStats;
    std::map<std::string, long long> currentLogConnectionStats;
    
    std::atomic<long long> connectionCount{0};
    
    // 禁止拷贝
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
};