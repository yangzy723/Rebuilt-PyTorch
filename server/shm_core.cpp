#include "ipc.h"
#include "logger.h"
#include "shm_core.h"
#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <csignal>
#include <sstream>

// ======================= ShmChannel 实现 =======================

ShmChannel::ShmChannel(ClientChannelStruct* ptr, std::string name, std::string type, std::string id, pid_t pid)
    : channelPtr(ptr), shmName(name), clientType(type), uniqueId(id), clientPid(pid) {}

ShmChannel::~ShmChannel() {
    if (channelPtr) {
        channelPtr->scheduler_ready.store(false, std::memory_order_release);
        munmap(channelPtr, sizeof(ClientChannelStruct));
    }
}

void ShmChannel::unlink() {
    shm_unlink(shmName.c_str());
}

void ShmChannel::setReady() {
    if (channelPtr) channelPtr->scheduler_ready.store(true, std::memory_order_release);
}

bool ShmChannel::isConnected() {
    if (!channelPtr) return false;
    // 检查连接标志
    if (!channelPtr->client_connected.load(std::memory_order_acquire)) return false;
    // 检查 PID (可选，保留原逻辑)
    if (clientPid > 0 && kill(clientPid, 0) != 0 && errno != EPERM) return false;
    return true;
}

bool ShmChannel::spsc_try_pop(char* out_data, size_t max_len) {
    auto& q = channelPtr->request_queue;
    uint64_t head = q.head.load(std::memory_order_relaxed);
    if (head == q.tail.load(std::memory_order_acquire)) return false;

    size_t copy_len = strlen(q.buffer[head]);
    if (copy_len >= max_len) copy_len = max_len - 1;
    memcpy(out_data, q.buffer[head], copy_len);
    out_data[copy_len] = '\0';

    q.head.store((head + 1) % SPSC_QUEUE_SIZE, std::memory_order_release);
    return true;
}

bool ShmChannel::spsc_try_push(const char* data, size_t len) {
    auto& q = channelPtr->response_queue;
    uint64_t tail = q.tail.load(std::memory_order_relaxed);
    uint64_t next_tail = (tail + 1) % SPSC_QUEUE_SIZE;
    if (next_tail == q.head.load(std::memory_order_acquire)) return false;

    size_t copy_len = (len < SPSC_MSG_SIZE - 1) ? len : (SPSC_MSG_SIZE - 1);
    memcpy(q.buffer[tail], data, copy_len);
    q.buffer[tail][copy_len] = '\0';

    q.tail.store(next_tail, std::memory_order_release);
    return true;
}

bool ShmChannel::recvBlocking(std::string& outMsg) {
    char buffer[SPSC_MSG_SIZE];
    // 忙等待实现，保留原有的性能特性
    while (!spsc_try_pop(buffer, SPSC_MSG_SIZE)) {
        if (!isConnected()) return false;
        __asm__ __volatile__("pause" ::: "memory");
    }
    outMsg = std::string(buffer);
    return true;
}

bool ShmChannel::sendBlocking(const std::string& msg) {
    // 简单的超时机制 (例如 5秒)
    int attempts = 0;
    while (!spsc_try_push(msg.c_str(), msg.length())) {
        if (attempts++ > 5000000) return false;
        __asm__ __volatile__("pause" ::: "memory");
    }
    return true;
}

// ======================= ShmServer 实现 =======================

std::string get_user_suffix() {
    const char* u = std::getenv("USER");
    return (u && *u) ? std::string("_") + u : "_nouser";
}

ShmServer::ShmServer() : running(false), registry(nullptr) {}

std::string ShmServer::getRegistryName() {
    return std::string("/kernel_scheduler_registry") + get_user_suffix();
}

bool ShmServer::init() {
    std::string name = getRegistryName();
    int fd = shm_open(name.c_str(), O_RDWR | O_CREAT, 0666);
    if (fd == -1) {
        perror("shm_open registry");
        return false;
    }
    if (ftruncate(fd, sizeof(ClientRegistry)) == -1) {
        perror("ftruncate registry");
        close(fd);
        return false;
    }
    void* ptr = mmap(nullptr, sizeof(ClientRegistry), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (ptr == MAP_FAILED) return false;

    registry = static_cast<ClientRegistry*>(ptr);
    registry->init();
    registry->scheduler_ready.store(true, std::memory_order_release);
    
    std::cout << "[ShmServer] Registry initialized: " << name << std::endl;
    return true;
}

ShmServer::~ShmServer() {
    stop();
    if (registry) {
        registry->scheduler_ready.store(false, std::memory_order_release);
        munmap(registry, sizeof(ClientRegistry));
        shm_unlink(getRegistryName().c_str());
    }
}

void ShmServer::start(std::function<void(std::unique_ptr<IChannel>)> onNewClient) {
    callback = onNewClient;
    running.store(true);
    scannerThread = std::thread(&ShmServer::scannerLoop, this);
}

void ShmServer::stop() {
    running.store(false);
    if (scannerThread.joinable()) scannerThread.join();
}

void ShmServer::scannerLoop() {
    uint32_t lastVersion = 0;
    while (running.load()) {
        if (!registry) { usleep(100000); continue; }

        uint32_t currentVersion = registry->version.load(std::memory_order_acquire);
        if (currentVersion != lastVersion) {
            for (size_t i = 0; i < MAX_REGISTERED_CLIENTS; i++) {
                if (registry->entries[i].active.load(std::memory_order_acquire)) {
                    discoverClient(i);
                }
            }
            lastVersion = currentVersion;
        }
        cleanupDisconnected(); // 维护注册表状态
        usleep(100000);
    }
}

void ShmServer::discoverClient(int slot) {
    std::lock_guard<std::mutex> lock(internalMutex);
    
    // 检查是否已经在服务 (简单检查，实际生产可能需要更复杂的映射)
    for (int s : activeSlots) { if (s == slot) return; }

    auto& entry = registry->entries[slot];
    std::string shmName(entry.shm_name);
    
    // 打开客户端通道
    int fd = shm_open(shmName.c_str(), O_RDWR, 0666);
    if (fd == -1) return; // 尚未准备好

    void* ptr = mmap(nullptr, sizeof(ClientChannelStruct), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    
    if (ptr != MAP_FAILED) {
        activeSlots.push_back(slot);
        
        auto channel = std::unique_ptr<IChannel>(new ShmChannel(
            static_cast<ClientChannelStruct*>(ptr),
            shmName,
            entry.client_type,
            entry.unique_id,
            static_cast<pid_t>(entry.client_pid)
        ));
        
        // 通知上层
        if (callback) callback(std::move(channel));
    }
}

void ShmServer::cleanupDisconnected() {
    std::lock_guard<std::mutex> lock(internalMutex);
    // 这里的逻辑稍微简化：只负责清理 registry 中的 active 标志
    // 实际的 Channel 对象生命周期由 Scheduler 管理，当 Scheduler 发现 Channel 断开时会释放对象
    // 但我们需要更新 activeSlots 列表。
    // 为了简单解耦，我们假设 Registry 只负责“生产”连接。
    // 清理 Registry 状态的逻辑保留在 ShmServer 中：
    
    auto it = activeSlots.begin();
    while (it != activeSlots.end()) {
        int slot = *it;
        bool stillActive = registry->entries[slot].active.load(std::memory_order_acquire);
        if (!stillActive) {
            it = activeSlots.erase(it);
        } else {
            ++it;
        }
    }
}