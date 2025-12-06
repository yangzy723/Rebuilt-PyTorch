#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <map>
#include <algorithm>
#include <csignal>

#include "IPCProtocol.h"

// ============================================================
//  全局变量
// ============================================================

std::mutex logMutex;           
std::ofstream globalLogFile;   
long long connectionCount = 0; 

std::atomic<long long> globalKernelId(0);
std::mutex statsMutex;
std::map<std::string, long long> currentLogKernelStats;

// 运行控制标志
std::atomic<bool> g_running(true);

// 共享内存通道
ClientChannel* g_pytorchChannel = nullptr;
ClientChannel* g_sglangChannel = nullptr;

// ============================================================
//  信号处理
// ============================================================

void signalHandler(int signum) {
    std::cout << "\n[Scheduler] 收到信号 " << signum << "，正在关闭..." << std::endl;
    g_running.store(false, std::memory_order_release);
}

// ============================================================
//  统计与日志函数
// ============================================================

// 注意：调用此函数时，调用者必须已经持有了 logMutex
void flushStatsAndReset() {
    std::lock_guard<std::mutex> lock(statsMutex); // 锁住统计数据

    if (!globalLogFile.is_open()) return;

    globalLogFile << "\n-------------------------------------------------------\n";
    globalLogFile << "      Kernel Statistics for this Log File\n";
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
}

std::string getCurrentTimeStrForFile() {
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time);
    std::stringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
    return ss.str();
}

// 注意：调用此函数时，必须在外部持有 logMutex，防止多线程同时写入/切换
void rotateLogFile() {
    // 1. 如果有旧文件，先写入统计信息，再关闭
    if (globalLogFile.is_open()) {
        flushStatsAndReset();
        globalLogFile.close();
        std::cout << "[Main] 上一轮日志统计已写入并关闭。" << std::endl;
    }

    // 2. 创建新文件
    std::string filename = "logs/" + getCurrentTimeStrForFile() + ".log";
    globalLogFile.open(filename, std::ios::out | std::ios::app);
    globalKernelId.store(0);
    if (!globalLogFile.is_open()) {
        std::cerr << "[Main] 致命错误: 无法创建日志文件 " << filename << std::endl;
    } else {
        std::cout << "[Main] 新的一轮开始，日志文件已创建: " << filename << std::endl;
    }
}

void writeLog(const std::string& message) {
    std::lock_guard<std::mutex> lock(logMutex); 
    if (globalLogFile.is_open()) {
        globalLogFile << message << std::endl;
        globalLogFile.flush(); 
    }
}

void recordKernelStat(const std::string& kernelType) {
    std::lock_guard<std::mutex> lock(statsMutex);
    currentLogKernelStats[kernelType]++;
}

// ============================================================
//  业务逻辑
// ============================================================

std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

std::pair<bool, std::string> makeDecision(const std::string& kernelType) {
    return {true, "OK"};
}

// ============================================================
//  处理单个客户端通道的线程函数
// ============================================================

void serviceClientChannel(ClientChannel* channel, const std::string& clientName) {
    std::stringstream ss;
    ss << "[Scheduler] 开始监听 " << clientName << " 通道";
    writeLog(ss.str());
    std::cout << ss.str() << std::endl;

    // 标记调度器已准备好
    channel->scheduler_ready.store(true, std::memory_order_release);

    char buffer[SPSC_MSG_SIZE];

    while (g_running.load(std::memory_order_acquire)) {
        // 非阻塞尝试读取请求，使用忙等待+退避策略
        int spin_count = 0;
        while (!channel->request_queue.try_pop(buffer, SPSC_MSG_SIZE)) {
            if (!g_running.load(std::memory_order_acquire)) {
                break;
            }
            // 忙等待：前 1000 次快速轮询，然后逐渐增加延迟
            // if (spin_count < 1000) {
            //     // CPU pause hint，减少功耗
            //     __asm__ __volatile__("pause" ::: "memory");
            //     spin_count++;
            // } else if (spin_count < 10000) {
            //     // 每 100 次检查一次
            //     if (spin_count % 100 == 0) {
            //         usleep(1);  // 1 微秒
            //     }
            //     spin_count++;
            // } else {
            //     // 长时间等待，使用更长的休眠
            //     usleep(10);  // 10 微秒
            //     spin_count = 0;  // 重置计数器
            // }
        }
        
        if (!g_running.load(std::memory_order_acquire)) {
            break;
        }

        // 直接处理 buffer，避免不必要的字符串拷贝
        size_t msg_len = strlen(buffer);
        // 去除尾部换行符
        while (msg_len > 0 && (buffer[msg_len - 1] == '\n' || buffer[msg_len - 1] == '\r')) {
            buffer[msg_len - 1] = '\0';
            msg_len--;
        }
        
        std::string message(buffer, msg_len);
        auto parts = split(message, '|');
        if (parts.size() < 3) {
            writeLog("[Scheduler] 格式错误 (" + message + ")");
            continue;
        }
        
        std::string kernelType = parts[0];     
        std::string reqId = parts[1];        
        std::string source = parts[2];       

        long long currentId = ++globalKernelId;

        recordKernelStat(kernelType);

        // 减少日志写入频率：每 100 个内核记录一次，或关键事件
        if (currentId % 100 == 0 || currentId <= 10) {
            ss.str("");
            ss << "Kernel " << currentId << " arrived: " << kernelType << "|" << reqId << " from " << source;
            writeLog(ss.str());
        }

        auto decision = makeDecision(kernelType);
        bool allowed = decision.first;
        std::string reason = decision.second;

        std::string response = createResponseMessage(reqId, allowed, reason);
        
        // 发送响应到响应队列
        if (!channel->response_queue.push_blocking(response, 5000)) {
            writeLog("[Scheduler] 发送响应超时: " + clientName);
        }
    }

    ss.str("");
    ss << "[Scheduler] " << clientName << " 通道处理线程退出";
    writeLog(ss.str());
    std::cout << ss.str() << std::endl;
}

// ============================================================
//  初始化共享内存
// ============================================================

bool initSharedMemory() {
    std::cout << "[Scheduler] 正在初始化共享内存..." << std::endl;

    // 创建 PyTorch 通道
    g_pytorchChannel = SharedMemoryHelper::create_or_open(SHM_NAME_PYTORCH, true);
    if (!g_pytorchChannel) {
        std::cerr << "[Scheduler] 创建 PyTorch 共享内存失败" << std::endl;
        return false;
    }
    std::cout << "[Scheduler] PyTorch 共享内存通道已创建: " << SHM_NAME_PYTORCH << std::endl;

    // 创建 SGLang 通道
    g_sglangChannel = SharedMemoryHelper::create_or_open(SHM_NAME_SGLANG, true);
    if (!g_sglangChannel) {
        std::cerr << "[Scheduler] 创建 SGLang 共享内存失败" << std::endl;
        SharedMemoryHelper::unmap(g_pytorchChannel);
        SharedMemoryHelper::unlink(SHM_NAME_PYTORCH);
        return false;
    }
    std::cout << "[Scheduler] SGLang 共享内存通道已创建: " << SHM_NAME_SGLANG << std::endl;

    return true;
}

// ============================================================
//  清理共享内存
// ============================================================

void cleanupSharedMemory() {
    std::cout << "[Scheduler] 正在清理共享内存..." << std::endl;

    if (g_pytorchChannel) {
        g_pytorchChannel->scheduler_ready.store(false, std::memory_order_release);
        SharedMemoryHelper::unmap(g_pytorchChannel);
        SharedMemoryHelper::unlink(SHM_NAME_PYTORCH);
        g_pytorchChannel = nullptr;
        std::cout << "[Scheduler] PyTorch 共享内存已清理" << std::endl;
    }

    if (g_sglangChannel) {
        g_sglangChannel->scheduler_ready.store(false, std::memory_order_release);
        SharedMemoryHelper::unmap(g_sglangChannel);
        SharedMemoryHelper::unlink(SHM_NAME_SGLANG);
        g_sglangChannel = nullptr;
        std::cout << "[Scheduler] SGLang 共享内存已清理" << std::endl;
    }
}

// ============================================================
//  主函数
// ============================================================

int main() {
    // 注册信号处理
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    mkdir("logs", 0777);

    // 初始化共享内存
    if (!initSharedMemory()) {
        std::cerr << "[Scheduler] 共享内存初始化失败，退出" << std::endl;
        return EXIT_FAILURE;
    }

    // 创建初始日志文件
    {
        std::lock_guard<std::mutex> lock(logMutex);
        rotateLogFile();
    }

    std::cout << "[Scheduler] 服务端运行中 (使用 SHM SPSC 通信)..." << std::endl;
    std::cout << "[Scheduler] PyTorch 通道: " << SHM_NAME_PYTORCH << std::endl;
    std::cout << "[Scheduler] SGLang 通道: " << SHM_NAME_SGLANG << std::endl;

    // 启动客户端处理线程
    std::thread pytorchThread(serviceClientChannel, g_pytorchChannel, "PyTorch");
    std::thread sglangThread(serviceClientChannel, g_sglangChannel, "SGLang");

    // 主线程等待退出信号
    while (g_running.load(std::memory_order_acquire)) {
        sleep(1);
        
        // 定期轮转日志（每分钟）
        static auto lastRotate = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::minutes>(now - lastRotate).count() >= 1) {
            std::lock_guard<std::mutex> lock(logMutex);
            rotateLogFile();
            lastRotate = now;
        }
    }

    std::cout << "[Scheduler] 正在等待工作线程退出..." << std::endl;

    // 等待线程退出
    if (pytorchThread.joinable()) pytorchThread.join();
    if (sglangThread.joinable()) sglangThread.join();

    // 写入最终统计并关闭日志
    {
        std::lock_guard<std::mutex> lock(logMutex);
        if (globalLogFile.is_open()) {
            flushStatsAndReset();
            globalLogFile.close();
        }
    }

    // 清理共享内存
    cleanupSharedMemory();

    std::cout << "[Scheduler] 已安全退出" << std::endl;
    return 0;
}
