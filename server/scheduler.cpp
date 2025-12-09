#include "scheduler.h"
#include "logger.h"
#include <sstream>
#include <iostream>

Scheduler::Scheduler() {}

Scheduler::~Scheduler() {
    stop();
}

void Scheduler::stop() {
    running = false;
    std::lock_guard<std::mutex> lock(threadsMutex);
    for (auto& t : workers) {
        if (t.joinable()) t.join();
    }
    workers.clear();
}

size_t Scheduler::getActiveCount() {
    std::lock_guard<std::mutex> lock(threadsMutex);
    return workers.size(); 
}

std::pair<bool, std::string> Scheduler::makeDecision(const std::string& kernelType) {
    // 核心调度算法
    return {true, "OK"};
}

// 辅助函数：分割字符串
std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

void Scheduler::onNewClient(std::unique_ptr<IChannel> channel) {
    std::lock_guard<std::mutex> lock(threadsMutex);
    // 转移 channel 所有权给线程
    workers.emplace_back(&Scheduler::clientHandler, this, std::move(channel));
}

void Scheduler::clientHandler(std::unique_ptr<IChannel> channel) {
    long long sessionId = Logger::instance().incrementConnectionCount();
    std::string clientKey = channel->getType() + ":" + channel->getId();
    std::string logKey = channel->getName();
    Logger::instance().recordConnectionStat(clientKey);

    std::stringstream ss;
    ss << "[Scheduler] Session #" << sessionId << " started for " 
       << clientKey << " (SHM: " << channel->getName() << ")";
    Logger::instance().write(ss.str(), logKey);
    std::cout << ss.str() << std::endl;

    channel->setReady();

    std::string message;
    while (running && channel->isConnected()) {
        // 阻塞接收 (底层实现忙等待)
        if (!channel->recvBlocking(message)) {
             continue; 
        }

        // 简单的协议解析
        while (!message.empty() && (message.back() == '\n' || message.back() == '\r')) {
            message.pop_back();
        }

        auto parts = split(message, '|');
        if (parts.size() < 3) {
            continue;
        }

        std::string kernelType = parts[0];     
        std::string reqId = parts[1];        
        
        long long currentId = ++globalKernelId;
        Logger::instance().recordKernelStat(kernelType);

        // 抽样日志
        if (currentId % 100 == 0 || currentId <= 10) {
            ss.str("");
            ss << "Kernel " << currentId << ": " << kernelType;
            Logger::instance().write(ss.str(), logKey);
        }

        // 决策
        auto decision = makeDecision(kernelType);
        
        // 构建响应
        std::string response = reqId + "|" + (decision.first ? "1" : "0") + "|" + decision.second + "\n";
        
        if (!channel->sendBlocking(response)) {
            Logger::instance().write("[Scheduler] Send timeout for " + clientKey, logKey);
        }
    }

    ss.str("");
    ss << "[Scheduler] Session #" << sessionId << " ended (" << clientKey << ")";
    Logger::instance().write(ss.str(), logKey);
    std::cout << ss.str() << std::endl;
}