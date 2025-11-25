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

#include "IPCProtocol.h"

// ---------------- 全局变量 ----------------

std::mutex logMutex;           
std::ofstream globalLogFile;   
long long connectionCount = 0; 

std::atomic<long long> globalKernelId(0);

// ---------------- 日志辅助函数 ----------------

std::string getCurrentTimeStrForFile() {
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time);
    std::stringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
    return ss.str();
}

void rotateLogFile() {
    if (globalLogFile.is_open()) {
        globalLogFile.close();
        std::cout << "[Main] 上一轮日志已关闭。" << std::endl;
    }
    std::string filename = "logs/" + getCurrentTimeStrForFile() + ".log";
    globalLogFile.open(filename, std::ios::out | std::ios::app);
    
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
    } else {
        std::cout << "[Log Lost]: " << message << std::endl;
    }
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
    // 可以在这里根据 kernelType 做具体策略
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
        // 清理换行符
        while (!message.empty() && (message.back() == '\n' || message.back() == '\r')) {
            message.pop_back();
        }

        // 客户端格式: kernel_type|req_id|source
        auto parts = split(message, '|');
        if (parts.size() < 3) {
            writeLog("[Scheduler] 格式错误 (" + message + ")，断开。");
            close(clientSocket);
            break;
        }
        
        // 1. 获取各个部分
        std::string kernelType = parts[0];     // e.g., "GemmInternalCublas"
        std::string reqId = parts[1];        // e.g., "req_5"
        std::string source = parts[2];       // e.g., "pytorch" or "slang"

        // 2. 增加全局 Kernel ID
        long long currentId = ++globalKernelId;

        // 3. 记录日志：Kernel {id} arrived: kernel_name|req_id from slang/pytorch
        ss.str("");
        ss << "Kernel " << currentId << " arrived: " << kernelType << "|" << reqId << " from " << source;
        writeLog(ss.str());

        // 4. 决策逻辑
        auto decision = makeDecision(kernelType);
        bool allowed = decision.first;
        std::string reason = decision.second;

        // 5. 发送响应 (协议: reqId|allowed|reason)
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
            std::lock_guard<std::mutex> lock(logMutex);
            if (connectionCount % 2 == 0) {
                rotateLogFile();
            }
            connectionCount++;
        }

        std::thread clientThread(serviceClient, new_socket);
        clientThread.detach(); 
    }

    if(globalLogFile.is_open()) globalLogFile.close();
    close(server_fd);
    return 0;
}