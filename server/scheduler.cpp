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
#include <memory>
#include <cstdlib>  // for atoi
#include <cerrno>   // for errno
#include <sys/types.h>  // for pid_t

#include "IPCProtocol.h"

// ============================================================
//  全局变量
// ============================================================

std::mutex logMutex;           
std::ofstream globalLogFile;   
std::atomic<long long> connectionCount(0);  // 连接/会话计数（兼容socket版本）

std::atomic<long long> globalKernelId(0);
std::mutex statsMutex;
std::map<std::string, long long> currentLogKernelStats;
std::map<std::string, long long> currentLogConnectionStats;  // 每个客户端的连接统计

// 运行控制标志
std::atomic<bool> g_running(true);

// 注册表共享内存
ClientRegistry* g_registry = nullptr;

// ============================================================
//  动态客户端管理
// ============================================================

struct ActiveClient {
    int registrySlot;                      // 在注册表中的槽位
    std::string shmName;                   // 共享内存名称
    std::string clientType;                // 客户端类型
    std::string uniqueId;                  // 唯一标识
    pid_t clientPid;                       // 客户端进程 PID（用于检测进程存活）
    ClientChannel* channel;                // 通道指针
    std::thread serviceThread;             // 服务线程
    std::atomic<bool> running;             // 服务线程运行标志
    std::atomic<uint64_t> lastActivityTime; // 最后活动时间（毫秒）
    
    ActiveClient() : registrySlot(-1), clientPid(0), channel(nullptr), running(false), lastActivityTime(0) {}
    ~ActiveClient() {
        running.store(false, std::memory_order_release);
        if (serviceThread.joinable()) {
            serviceThread.join();
        }
        if (channel) {
            channel->scheduler_ready.store(false, std::memory_order_release);
            SharedMemoryHelper::unmap(channel);
            channel = nullptr;
        }
    }
};

std::mutex clientsMutex;
std::map<int, std::unique_ptr<ActiveClient>> g_activeClients;  // slot -> ActiveClient

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
    globalLogFile << "      Session Statistics (Compatible with Socket)\n";
    globalLogFile << "-------------------------------------------------------\n";
    globalLogFile << "Total Connections/Sessions: " << connectionCount.load() << "\n";

    // 输出每个客户端的连接统计
    if (!currentLogConnectionStats.empty()) {
        globalLogFile << "\nConnections by Client:\n";
        for (const auto& item : currentLogConnectionStats) {
            globalLogFile << "  " << item.first << ": " << item.second << " session(s)\n";
        }
    }

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
    currentLogConnectionStats.clear();
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

void serviceClientChannel(ActiveClient* client) {
    std::stringstream ss;
    
    // 记录连接/会话开始（兼容socket版本的连接计数逻辑）
    long long sessionId = connectionCount.fetch_add(1, std::memory_order_relaxed) + 1;
    {
        std::lock_guard<std::mutex> lock(statsMutex);
        currentLogConnectionStats[client->clientType + ":" + client->uniqueId]++;
    }
    
    ss << "[Scheduler] 会话 #" << sessionId << " 开始服务 " 
       << client->clientType << " 客户端 (ID: " << client->uniqueId 
       << ", SHM: " << client->shmName << ")";
    writeLog(ss.str());
    std::cout << ss.str() << std::endl;

    // 标记调度器已准备好服务此通道
    client->channel->scheduler_ready.store(true, std::memory_order_release);

    // 初始化活动时间
    auto getCurrentTimeMs = []() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count();
    };
    client->lastActivityTime.store(getCurrentTimeMs(), std::memory_order_release);

    char buffer[SPSC_MSG_SIZE];
    int spin_count = 0;

    while (g_running.load(std::memory_order_acquire) && 
           client->running.load(std::memory_order_acquire)) {
        
        // 忙等待读取请求（极低延迟模式）
        while (!client->channel->request_queue.try_pop(buffer, SPSC_MSG_SIZE)) {
            if (!g_running.load(std::memory_order_acquire) ||
                !client->running.load(std::memory_order_acquire)) {
                goto exit_loop;
            }
            
            // 每 10000 次循环检查一次客户端连接状态
            if (++spin_count >= 10000) {
                spin_count = 0;
                if (!client->channel->client_connected.load(std::memory_order_acquire)) {
                    ss.str("");
                    ss << "[Scheduler] 客户端已断开连接: " << client->shmName;
                    writeLog(ss.str());
                    std::cout << ss.str() << std::endl;
                    goto exit_loop;
                }
            }
            
            // CPU pause hint，减少功耗和总线争用
            __asm__ __volatile__("pause" ::: "memory");
        }

        // 更新最后活动时间
        client->lastActivityTime.store(getCurrentTimeMs(), std::memory_order_release);

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
        std::string uniqueId = (parts.size() >= 4) ? parts[3] : "";

        long long currentId = ++globalKernelId;

        recordKernelStat(kernelType);

        // 减少日志写入频率：每 100 个内核记录一次，或关键事件
        if (currentId % 100 == 0 || currentId <= 10) {
            ss.str("");
            ss << "Kernel " << currentId << " arrived: " << kernelType << "|" << reqId 
               << " from " << source;
            if (!uniqueId.empty()) {
                ss << " (UNIQUE_ID: " << uniqueId << ")";
            }
            writeLog(ss.str());
        }

        auto decision = makeDecision(kernelType);
        bool allowed = decision.first;
        std::string reason = decision.second;

        std::string response = createResponseMessage(reqId, allowed, reason);
        
        // 发送响应到响应队列
        if (!client->channel->response_queue.push_blocking(response, 5000)) {
            writeLog("[Scheduler] 发送响应超时: " + client->shmName);
        }
    }

exit_loop:
    ss.str("");
    ss << "[Scheduler] 会话 #" << sessionId << " " << client->clientType 
       << " 客户端服务线程退出 (ID: " << client->uniqueId << ")";
    writeLog(ss.str());
    std::cout << ss.str() << std::endl;
}

// ============================================================
//  发现并服务新客户端
// ============================================================

void discoverAndServiceNewClient(int slot) {
    if (!g_registry) return;
    
    ClientRegistryEntry& entry = g_registry->entries[slot];
    
    // 获取客户端信息（在锁外读取，因为只有客户端会写入这些字段）
    std::string shmName(entry.shm_name);
    std::string clientType(entry.client_type);
    std::string uniqueId(entry.unique_id);
    pid_t clientPid = static_cast<pid_t>(entry.client_pid.load(std::memory_order_acquire));
    
    // 使用锁保护整个检查和创建过程，防止竞态条件
    std::lock_guard<std::mutex> lock(clientsMutex);
    
    // 检查是否已经在服务此 slot
    if (g_activeClients.find(slot) != g_activeClients.end()) {
        return;  // 已经在服务
    }
    
    // 检查是否已经有其他线程在服务相同的共享内存通道（防止重复服务同一个 shmName）
    for (const auto& pair : g_activeClients) {
        if (pair.second && pair.second->shmName == shmName) {
            return;  // 已经有线程在服务此共享内存
        }
    }
    
    std::cout << "[Scheduler] 发现新客户端: " << clientType << " (ID: " << uniqueId 
              << ", PID: " << clientPid << ", SHM: " << shmName << ")" << std::endl;
    
    // 打开客户端的共享内存通道
    ClientChannel* channel = SharedMemoryHelper::create_or_open(shmName.c_str(), false);
    if (!channel) {
        std::cerr << "[Scheduler] 无法打开客户端共享内存: " << shmName << std::endl;
        return;
    }
    
    // 创建 ActiveClient (使用 C++11 兼容写法)
    std::unique_ptr<ActiveClient> client(new ActiveClient());
    client->registrySlot = slot;
    client->shmName = shmName;
    client->clientType = clientType;
    client->uniqueId = uniqueId;
    client->clientPid = clientPid;
    client->channel = channel;
    client->running.store(true, std::memory_order_release);
    
    // 先加入活跃客户端列表（防止其他线程重复创建）
    ActiveClient* clientPtr = client.get();
    g_activeClients[slot] = std::move(client);
    
    // 启动服务线程（线程会立即开始运行，但此时已经在 map 中了）
    g_activeClients[slot]->serviceThread = std::thread(serviceClientChannel, clientPtr);
}

// ============================================================
//  清理断开连接的客户端
// ============================================================

// 检测进程是否存在
static bool isProcessAlive(pid_t pid) {
    if (pid <= 0) return true;  // 无法确定，假设存活
    // kill(pid, 0) 不发送信号，只检测进程是否存在
    return (kill(pid, 0) == 0 || errno == EPERM);
}

void cleanupDisconnectedClients() {
    std::lock_guard<std::mutex> lock(clientsMutex);
    
    std::vector<int> toRemove;
    
    for (auto& pair : g_activeClients) {
        int slot = pair.first;
        ActiveClient* client = pair.second.get();
        
        // 检查客户端是否仍然活跃（在注册表中）
        bool stillActive = false;
        if (g_registry && slot < static_cast<int>(MAX_REGISTERED_CLIENTS)) {
            stillActive = g_registry->entries[slot].active.load(std::memory_order_acquire);
        }
        
        // 检查客户端连接状态
        bool stillConnected = client->channel && 
                              client->channel->client_connected.load(std::memory_order_acquire);
        
        // 检查客户端进程是否仍然存活（最可靠的检测方式）
        bool processAlive = isProcessAlive(client->clientPid);
        
        if (!stillActive || !stillConnected || !processAlive) {
            if (!processAlive) {
                std::cout << "[Scheduler] 检测到客户端进程已终止 (PID: " << client->clientPid 
                          << "): " << client->shmName << std::endl;
            } else {
                std::cout << "[Scheduler] 清理断开的客户端: " << client->shmName << std::endl;
            }
            client->running.store(false, std::memory_order_release);
            toRemove.push_back(slot);
        }
    }
    
    // 移除断开的客户端
    for (int slot : toRemove) {
        auto it = g_activeClients.find(slot);
        if (it != g_activeClients.end()) {
            ActiveClient* client = it->second.get();
            
            // 清理注册表条目（如果客户端没有正常注销）
            if (g_registry && slot < static_cast<int>(MAX_REGISTERED_CLIENTS)) {
                g_registry->entries[slot].active.store(false, std::memory_order_release);
            }
            
            // 尝试删除共享内存文件（客户端可能没来得及清理）
            if (!client->shmName.empty()) {
                SharedMemoryHelper::unlink(client->shmName.c_str());
            }
            
            g_activeClients.erase(it);
        }
    }
}

// ============================================================
//  注册表扫描线程
// ============================================================

void registryScannerThread() {
    std::cout << "[Scheduler] 注册表扫描线程已启动" << std::endl;
    
    uint32_t lastVersion = 0;
    
    while (g_running.load(std::memory_order_acquire)) {
        if (!g_registry) {
            usleep(100000);  // 100ms
            continue;
        }
        
        // 检查版本号是否变化
        uint32_t currentVersion = g_registry->version.load(std::memory_order_acquire);
        
        if (currentVersion != lastVersion) {
            // 有变化，扫描注册表
            for (size_t i = 0; i < MAX_REGISTERED_CLIENTS; i++) {
                if (g_registry->entries[i].active.load(std::memory_order_acquire)) {
                    discoverAndServiceNewClient(static_cast<int>(i));
                }
            }
            lastVersion = currentVersion;
        }
        
        // 定期清理断开连接的客户端
        cleanupDisconnectedClients();
        
        usleep(100000);  // 100ms 扫描间隔
    }
    
    std::cout << "[Scheduler] 注册表扫描线程已退出" << std::endl;
}

// ============================================================
//  初始化共享内存
// ============================================================

bool initSharedMemory() {
    std::cout << "[Scheduler] 正在初始化共享内存..." << std::endl;

    // 创建注册表共享内存
    g_registry = SharedMemoryHelper::create_or_open_registry(true);
    if (!g_registry) {
        std::cerr << "[Scheduler] 创建注册表共享内存失败" << std::endl;
        return false;
    }
    std::cout << "[Scheduler] 注册表共享内存已创建: " << get_registry_name() << std::endl;

    // 标记调度器已准备好（客户端可以开始注册）
    g_registry->scheduler_ready.store(true, std::memory_order_release);

    return true;
}

// ============================================================
//  清理共享内存
// ============================================================

void cleanupSharedMemory() {
    std::cout << "[Scheduler] 正在清理共享内存..." << std::endl;

    // 先清理所有活跃客户端
    {
        std::lock_guard<std::mutex> lock(clientsMutex);
        for (auto& pair : g_activeClients) {
            pair.second->running.store(false, std::memory_order_release);
        }
        g_activeClients.clear();
    }

    // 清理注册表
    if (g_registry) {
        g_registry->scheduler_ready.store(false, std::memory_order_release);
        SharedMemoryHelper::unmap_registry(g_registry);
        SharedMemoryHelper::unlink_registry();
        g_registry = nullptr;
        std::cout << "[Scheduler] 注册表共享内存已清理" << std::endl;
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

    std::cout << "[Scheduler] 服务端运行中 (动态多客户端模式)..." << std::endl;
    std::cout << "[Scheduler] 注册表: " << get_registry_name() << std::endl;
    std::cout << "[Scheduler] 等待客户端注册..." << std::endl;

    // 启动注册表扫描线程
    std::thread scannerThread(registryScannerThread);

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
        
        // 打印当前活跃客户端数量
        static auto lastStatus = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - lastStatus).count() >= 10) {
            std::lock_guard<std::mutex> lock(clientsMutex);
            std::cout << "[Scheduler] 当前活跃客户端数: " << g_activeClients.size() << std::endl;
            lastStatus = now;
        }
    }

    std::cout << "[Scheduler] 正在等待工作线程退出..." << std::endl;

    // 等待扫描线程退出
    if (scannerThread.joinable()) {
        scannerThread.join();
    }

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
