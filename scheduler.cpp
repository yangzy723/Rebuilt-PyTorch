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
#include <sys/socket.h>
#include <netinet/in.h>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <map>
#include <algorithm>

#include "IPCProtocol.h"

// ---------------- 全局变量 ----------------

std::mutex logMutex;           
std::ofstream globalLogFile;   
long long connectionCount = 0; 

std::atomic<long long> globalKernelId(0);
std::mutex statsMutex;
std::map<std::string, long long> currentLogKernelStats;

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

// ---------------- 日志辅助函数 ----------------

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

// ---------------- 业务逻辑 ----------------

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

void serviceClient(int clientSocket) {
    std::stringstream ss;
    ss << "[Scheduler] 收到连接 (Socket: " << clientSocket << ")";
    writeLog(ss.str()); 

    char buffer[1024] = {0};
    
    while (true) {
        memset(buffer, 0, 1024);
        ssize_t bytesRead = read(clientSocket, buffer, 1023);

        if (bytesRead <= 0) {
            ss.str(""); 
            if (bytesRead == 0) {
                ss << "[Scheduler] Socket " << clientSocket << " 已断开。";
            } else {
                ss << "[Scheduler] Socket " << clientSocket << " 读取错误。";
            }
            writeLog(ss.str());
            close(clientSocket);
            break;              
        }

        std::string message(buffer, bytesRead);
        while (!message.empty() && (message.back() == '\n' || message.back() == '\r')) {
            message.pop_back();
        }

        auto parts = split(message, '|');
        if (parts.size() < 3) {
            writeLog("[Scheduler] 格式错误 (" + message + ")，断开。");
            close(clientSocket);
            break;
        }
        
        std::string kernelType = parts[0];     
        std::string reqId = parts[1];        
        std::string source = parts[2];       

        long long currentId = ++globalKernelId;

        recordKernelStat(kernelType);

        ss.str("");
        ss << "Kernel " << currentId << " arrived: " << kernelType << "|" << reqId << " from " << source;
        writeLog(ss.str());

        auto decision = makeDecision(kernelType);
        bool allowed = decision.first;
        std::string reason = decision.second;

        std::string response = createResponseMessage(reqId, allowed, reason);
        if (send(clientSocket, response.c_str(), response.length(), 0) < 0) {
             writeLog("[Scheduler] 发送响应失败，连接断开。");
             close(clientSocket);
             break;
        }
    }
}

int main() {
    mkdir("logs", 0777);

    int server_fd;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(SCHEDULER_PORT);

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    if (listen(server_fd, 10) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }

    std::cout << "[Scheduler] 服务端运行中 (Port " << SCHEDULER_PORT << ")... " << std::endl;

    while (true) {
        int new_socket;
        if ((new_socket = accept(server_fd, (struct sockaddr*)&address, (socklen_t*)&addrlen)) < 0) {
            perror("accept");
            continue;
        }
        
        {
            // 在检查是否需要轮转日志前，必须持有 logMutex。
            // 这样可以确保在轮转（写入统计、关闭旧文件）的过程中，
            // 没有任何工作线程能通过 writeLog 写入日志，避免数据竞态或写入已关闭的文件。
            std::lock_guard<std::mutex> lock(logMutex);
            
            if (connectionCount % 2 == 0) {
                rotateLogFile();
            }
            connectionCount++;
        }

        std::thread clientThread(serviceClient, new_socket);
        clientThread.detach(); 
    }

    // 程序正常退出前的清理
    if(globalLogFile.is_open()) {
        std::lock_guard<std::mutex> lock(logMutex);
        flushStatsAndReset();
        globalLogFile.close();
    }
    
    close(server_fd);
    return 0;
}