#pragma once

/**
 * @file shm_transport.h
 * @brief 共享内存传输层实现
 * 
 * 此文件实现了基于 POSIX 共享内存的 IPC 传输层。
 * 客户端和服务端都可以使用此实现。
 */

#include "ipc_interface.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <csignal>
#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>

namespace ipc {
namespace shm {

// ============================================================
//  共享内存消息队列实现
// ============================================================

class ShmMessageQueue : public IMessageQueue {
public:
    explicit ShmMessageQueue(SPSCQueueData* data) : data_(data) {}
    ~ShmMessageQueue() override = default;

    bool trySend(const std::string& message) override {
        return trySend(message.c_str(), message.length());
    }

    bool trySend(const char* data, size_t len) override {
        return data_->tryPush(data, len);
    }

    bool sendBlocking(const std::string& message, int timeout_ms) override {
        return sendBlocking(message.c_str(), message.length(), timeout_ms);
    }

    bool sendBlocking(const char* data, size_t len, int timeout_ms) override {
        return data_->pushBlocking(data, len, timeout_ms);
    }

    bool tryReceive(char* buffer, size_t buffer_size) override {
        return data_->tryPop(buffer, buffer_size);
    }

    bool tryReceive(std::string& out_message) override {
        char buf[MAX_MSG_SIZE];
        if (data_->tryPop(buf, MAX_MSG_SIZE)) {
            out_message = buf;
            return true;
        }
        return false;
    }

    bool receiveBlocking(char* buffer, size_t buffer_size, int timeout_ms) override {
        return data_->popBlocking(buffer, buffer_size, timeout_ms);
    }

    bool receiveBlocking(std::string& out_message, int timeout_ms) override {
        char buf[MAX_MSG_SIZE];
        if (data_->popBlocking(buf, MAX_MSG_SIZE, timeout_ms)) {
            out_message = buf;
            return true;
        }
        return false;
    }

    bool isEmpty() const override {
        return data_->empty();
    }

    size_t size() const override {
        return data_->size();
    }

private:
    SPSCQueueData* data_;
};

// ============================================================
//  共享内存通道实现
// ============================================================

class ShmChannel : public IChannel {
public:
    ShmChannel(const std::string& name, ChannelData* data, bool ownsMemory,
               const std::string& clientType = "", const std::string& uniqueId = "", pid_t pid = 0)
        : name_(name)
        , data_(data)
        , ownsMemory_(ownsMemory)
        , clientType_(clientType)
        , uniqueId_(uniqueId)
        , clientPid_(pid)
        , requestQueue_(&data->request_queue)
        , responseQueue_(&data->response_queue) {}
    
    ~ShmChannel() override {
        if (data_) {
            munmap(data_, sizeof(ChannelData));
            data_ = nullptr;
        }
    }

    IMessageQueue& getRequestQueue() override { return requestQueue_; }
    IMessageQueue& getResponseQueue() override { return responseQueue_; }

    bool isClientConnected() const override {
        if (!data_) return false;
        if (!data_->client_connected.load(std::memory_order_acquire)) return false;
        // 可选：检查进程是否存活
        if (clientPid_ > 0 && kill(clientPid_, 0) != 0 && errno != EPERM) return false;
        return true;
    }

    void setClientConnected(bool connected) override {
        if (data_) data_->client_connected.store(connected, std::memory_order_release);
    }

    bool isServerReady() const override {
        return data_ ? data_->server_ready.load(std::memory_order_acquire) : false;
    }

    void setServerReady(bool ready) override {
        if (data_) data_->server_ready.store(ready, std::memory_order_release);
    }

    std::string getName() const override { return name_; }
    std::string getClientType() const override { return clientType_; }
    std::string getUniqueId() const override { return uniqueId_; }
    pid_t getClientPid() const override { return clientPid_; }

    // 获取原始数据指针（用于特殊用途）
    ChannelData* getRawData() { return data_; }

private:
    std::string name_;
    ChannelData* data_;
    bool ownsMemory_;
    std::string clientType_;
    std::string uniqueId_;
    pid_t clientPid_;
    ShmMessageQueue requestQueue_;
    ShmMessageQueue responseQueue_;
};

// ============================================================
//  共享内存注册表实现
// ============================================================

class ShmRegistry : public IRegistry {
public:
    ShmRegistry(RegistryData* data, bool ownsMemory)
        : data_(data), ownsMemory_(ownsMemory) {}
    
    ~ShmRegistry() override {
        if (data_) {
            munmap(data_, sizeof(RegistryData));
            data_ = nullptr;
        }
    }

    bool isServerReady() const override {
        return data_->server_ready.load(std::memory_order_acquire);
    }

    void setServerReady(bool ready) override {
        data_->server_ready.store(ready, std::memory_order_release);
    }

    int registerClient(const std::string& channelName,
                       const std::string& clientType,
                       const std::string& uniqueId,
                       int64_t pid) override {
        return data_->registerClient(channelName.c_str(), clientType.c_str(), 
                                     uniqueId.c_str(), pid);
    }

    void unregisterClient(int slot) override {
        data_->unregisterClient(slot);
    }

    void updateHeartbeat(int slot) override {
        data_->updateHeartbeat(slot);
    }

    bool getClientInfo(int slot, ClientInfo& outInfo) const override {
        if (slot < 0 || slot >= static_cast<int>(MAX_CLIENTS)) {
            return false;
        }
        const auto& entry = data_->entries[slot];
        outInfo.slot = slot;
        outInfo.active = entry.active.load(std::memory_order_acquire);
        outInfo.channelName = entry.channel_name;
        outInfo.clientType = entry.client_type;
        outInfo.uniqueId = entry.unique_id;
        outInfo.pid = entry.client_pid.load(std::memory_order_acquire);
        outInfo.lastHeartbeat = entry.last_heartbeat.load(std::memory_order_acquire);
        return true;
    }

    std::vector<ClientInfo> getActiveClients() const override {
        std::vector<ClientInfo> clients;
        for (size_t i = 0; i < MAX_CLIENTS; i++) {
            if (data_->entries[i].active.load(std::memory_order_acquire)) {
                ClientInfo info;
                getClientInfo(static_cast<int>(i), info);
                clients.push_back(info);
            }
        }
        return clients;
    }

    uint32_t getVersion() const override {
        return data_->version.load(std::memory_order_acquire);
    }

    // 获取原始数据指针
    RegistryData* getRawData() { return data_; }

private:
    RegistryData* data_;
    bool ownsMemory_;
};

// ============================================================
//  共享内存工厂实现
// ============================================================

class ShmTransportFactory : public ITransportFactory {
public:
    ~ShmTransportFactory() override = default;

    std::unique_ptr<IChannel> createChannel(const std::string& name, bool isCreator) override {
        int flags = O_RDWR;
        if (isCreator) {
            flags |= O_CREAT;
        }
        
        int fd = shm_open(name.c_str(), flags, 0666);
        if (fd == -1) {
            std::cerr << "[ShmTransport] Failed to open shared memory: " << name << std::endl;
            return nullptr;
        }

        if (isCreator) {
            if (ftruncate(fd, sizeof(ChannelData)) == -1) {
                std::cerr << "[ShmTransport] Failed to set shared memory size" << std::endl;
                close(fd);
                return nullptr;
            }
        }

        void* ptr = mmap(nullptr, sizeof(ChannelData), 
                        PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);

        if (ptr == MAP_FAILED) {
            std::cerr << "[ShmTransport] Failed to map shared memory" << std::endl;
            return nullptr;
        }

        ChannelData* data = static_cast<ChannelData*>(ptr);
        
        if (isCreator) {
            data->init();
        }

        return std::make_unique<ShmChannel>(name, data, isCreator);
    }

    /**
     * 创建通道（带客户端元信息，服务端使用）
     */
    std::unique_ptr<IChannel> createChannelWithInfo(const std::string& name, bool isCreator,
                                                     const std::string& clientType,
                                                     const std::string& uniqueId,
                                                     pid_t pid) {
        int flags = O_RDWR;
        if (isCreator) {
            flags |= O_CREAT;
        }
        
        int fd = shm_open(name.c_str(), flags, 0666);
        if (fd == -1) {
            return nullptr;
        }

        if (isCreator) {
            if (ftruncate(fd, sizeof(ChannelData)) == -1) {
                close(fd);
                return nullptr;
            }
        }

        void* ptr = mmap(nullptr, sizeof(ChannelData), 
                        PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);

        if (ptr == MAP_FAILED) {
            return nullptr;
        }

        ChannelData* data = static_cast<ChannelData*>(ptr);
        
        if (isCreator) {
            data->init();
        }

        return std::make_unique<ShmChannel>(name, data, isCreator, clientType, uniqueId, pid);
    }

    std::unique_ptr<IRegistry> createRegistry(bool isCreator) override {
        int flags = O_RDWR;
        if (isCreator) {
            flags |= O_CREAT;
        }
        
        std::string reg_name = getRegistryName();
        int fd = shm_open(reg_name.c_str(), flags, 0666);
        if (fd == -1) {
            std::cerr << "[ShmTransport] Failed to open registry shared memory: " << reg_name << std::endl;
            return nullptr;
        }

        if (isCreator) {
            if (ftruncate(fd, sizeof(RegistryData)) == -1) {
                std::cerr << "[ShmTransport] Failed to set registry shared memory size" << std::endl;
                close(fd);
                return nullptr;
            }
        }

        void* ptr = mmap(nullptr, sizeof(RegistryData), 
                        PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);

        if (ptr == MAP_FAILED) {
            std::cerr << "[ShmTransport] Failed to map registry shared memory" << std::endl;
            return nullptr;
        }

        RegistryData* data = static_cast<RegistryData*>(ptr);
        
        if (isCreator) {
            data->init();
        }

        return std::make_unique<ShmRegistry>(data, isCreator);
    }

    void destroyChannel(const std::string& name) override {
        shm_unlink(name.c_str());
    }

    void destroyRegistry() override {
        std::string reg_name = getRegistryName();
        shm_unlink(reg_name.c_str());
    }

    std::string getName() const override {
        return "SharedMemory";
    }
};

// ============================================================
//  服务端监听器实现
// ============================================================

class ShmServerListener : public IServerListener {
public:
    ShmServerListener() : running_(false), registry_(nullptr) {}

    ~ShmServerListener() override {
        stop();
        // 清理所有已知的客户端通道
        cleanupAllClientChannels();
        if (registry_) {
            registry_->setServerReady(false);
            factory_.destroyRegistry();
        }
    }

    bool init() override {
        auto reg = factory_.createRegistry(true);
        if (!reg) {
            return false;
        }
        registry_ = std::move(reg);
        registry_->setServerReady(true);
        
        std::cout << "[ShmServerListener] Registry initialized: " << getRegistryName() << std::endl;
        return true;
    }

    void start(std::function<void(std::unique_ptr<IChannel>)> onNewClient) override {
        callback_ = onNewClient;
        running_.store(true);
        scannerThread_ = std::thread(&ShmServerListener::scannerLoop, this);
    }

    void stop() override {
        running_.store(false);
        if (scannerThread_.joinable()) {
            scannerThread_.join();
        }
    }

    bool isRunning() const override {
        return running_.load();
    }

    IRegistry* getRegistry() override {
        return registry_.get();
    }

private:
    void scannerLoop() {
        uint32_t lastVersion = 0;
        
        while (running_.load()) {
            if (!registry_) {
                usleep(100000);
                continue;
            }

            uint32_t currentVersion = registry_->getVersion();
            if (currentVersion != lastVersion) {
                // 版本变化，扫描新客户端
                auto clients = registry_->getActiveClients();
                for (const auto& client : clients) {
                    discoverClient(client);
                }
                lastVersion = currentVersion;
            }
            
            cleanupDisconnected();
            usleep(100000);  // 100ms
        }
    }

    void discoverClient(const ClientInfo& client) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // 检查是否已经在服务
        for (int slot : activeSlots_) {
            if (slot == client.slot) return;
        }

        // 打开客户端通道
        auto channel = factory_.createChannelWithInfo(
            client.channelName, false,
            client.clientType, client.uniqueId, static_cast<pid_t>(client.pid));
        
        if (!channel) {
            return;  // 通道尚未准备好
        }

        activeSlots_.push_back(client.slot);

        // 通知上层
        if (callback_) {
            callback_(std::move(channel));
        }
    }

    void cleanupDisconnected() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = activeSlots_.begin();
        while (it != activeSlots_.end()) {
            int slot = *it;
            ClientInfo info;
            if (registry_->getClientInfo(slot, info) && !info.active) {
                // 客户端已断开，清理其共享内存通道
                if (!info.channelName.empty()) {
                    factory_.destroyChannel(info.channelName);
                }
                it = activeSlots_.erase(it);
            } else {
                ++it;
            }
        }
    }

    void cleanupAllClientChannels() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (!registry_) return;
        
        // 遍历所有注册表条目，清理所有客户端通道
        for (size_t i = 0; i < MAX_CLIENTS; i++) {
            ClientInfo info;
            if (registry_->getClientInfo(static_cast<int>(i), info)) {
                if (!info.channelName.empty()) {
                    factory_.destroyChannel(info.channelName);
                }
            }
        }
        activeSlots_.clear();
    }

    std::atomic<bool> running_;
    ShmTransportFactory factory_;
    std::unique_ptr<IRegistry> registry_;
    std::thread scannerThread_;
    std::function<void(std::unique_ptr<IChannel>)> callback_;
    
    std::mutex mutex_;
    std::vector<int> activeSlots_;
};

// ============================================================
//  客户端连接管理实现
// ============================================================

class ShmClientConnection : public IClientConnection {
public:
    explicit ShmClientConnection(const std::string& channelName,
                                 const std::string& clientType = "pytorch",
                                 const std::string& uniqueId = "")
        : channelName_(channelName)
        , clientType_(clientType)
        , uniqueId_(uniqueId)
        , registrySlot_(-1)
        , connected_(false) {}

    ~ShmClientConnection() override {
        disconnect();
    }

    bool connect(int timeout_ms) override {
        if (connected_) return true;

        // 1. 打开注册表
        registry_ = factory_.createRegistry(false);
        if (!registry_) {
            std::cerr << "[ShmClientConnection] Failed to open registry, scheduler may not be running" << std::endl;
            return false;
        }

        // 2. 等待调度器就绪
        int waited = 0;
        const int interval = 100;
        while (!registry_->isServerReady()) {
            if (timeout_ms >= 0 && waited >= timeout_ms) {
                std::cerr << "[ShmClientConnection] Timeout waiting for scheduler to be ready" << std::endl;
                registry_.reset();
                return false;
            }
            usleep(interval * 1000);
            waited += interval;
        }

        // 3. 创建通道
        channel_ = factory_.createChannel(channelName_, true);
        if (!channel_) {
            std::cerr << "[ShmClientConnection] Failed to create channel: " << channelName_ << std::endl;
            registry_.reset();
            return false;
        }

        // 4. 注册客户端
        std::string uid = uniqueId_.empty() ? std::to_string(getpid()) : uniqueId_;
        registrySlot_ = registry_->registerClient(channelName_, clientType_, uid, getpid());
        if (registrySlot_ < 0) {
            std::cerr << "[ShmClientConnection] Registry is full" << std::endl;
            factory_.destroyChannel(channelName_);
            channel_.reset();
            registry_.reset();
            return false;
        }

        // 5. 标记客户端已连接
        channel_->setClientConnected(true);

        // 6. 等待服务端准备好
        waited = 0;
        while (!channel_->isServerReady()) {
            if (timeout_ms >= 0 && waited >= timeout_ms * 2) {
                std::cerr << "[ShmClientConnection] Timeout waiting for server to be ready" << std::endl;
                registry_->unregisterClient(registrySlot_);
                factory_.destroyChannel(channelName_);
                channel_.reset();
                registry_.reset();
                registrySlot_ = -1;
                return false;
            }
            usleep(interval * 1000);
            waited += interval;
        }

        connected_ = true;
        return true;
    }

    void disconnect() override {
        if (!connected_) return;

        if (registry_ && registrySlot_ >= 0) {
            registry_->unregisterClient(registrySlot_);
            registrySlot_ = -1;
        }

        if (channel_) {
            channel_->setClientConnected(false);
            factory_.destroyChannel(channelName_);
            channel_.reset();
        }

        registry_.reset();
        connected_ = false;
    }

    bool isConnected() const override {
        return connected_;
    }

    bool sendRequest(const std::string& request, 
                     std::string& response,
                     int timeout_ms) override {
        if (!connected_ || !channel_) {
            return false;
        }

        if (!channel_->getRequestQueue().sendBlocking(request, timeout_ms)) {
            return false;
        }

        return channel_->getResponseQueue().receiveBlocking(response, timeout_ms);
    }

    IChannel* getChannel() override {
        return channel_.get();
    }

private:
    std::string channelName_;
    std::string clientType_;
    std::string uniqueId_;
    ShmTransportFactory factory_;
    std::unique_ptr<IRegistry> registry_;
    std::unique_ptr<IChannel> channel_;
    int registrySlot_;
    bool connected_;
};

} // namespace shm
} // namespace ipc


