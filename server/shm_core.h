#pragma once
#include "ipc.h"
#include "config.h"
#include <atomic>
#include <thread>
#include <vector>
#include <mutex>

class ShmChannel : public IChannel {
public:
    ShmChannel(ClientChannelStruct* ptr, std::string name, std::string type, std::string id, pid_t pid);
    ~ShmChannel();

    bool recvBlocking(std::string& outMsg) override;
    bool sendBlocking(const std::string& msg) override;
    bool isConnected() override;
    void setReady() override;
    
    std::string getId() const override { return uniqueId; }
    std::string getType() const override { return clientType; }
    std::string getName() const override { return shmName; }

    // 清理
    void unlink();

private:
    ClientChannelStruct* channelPtr;
    std::string shmName;
    std::string clientType;
    std::string uniqueId;
    pid_t clientPid;

    // 辅助 SPSC 逻辑
    bool spsc_try_pop(char* out_data, size_t max_len);
    bool spsc_try_push(const char* data, size_t len);
};

class ShmServer : public IIPCServer {
public:
    ShmServer();
    ~ShmServer();

    bool init() override;
    void start(std::function<void(std::unique_ptr<IChannel>)> onNewClient) override;
    void stop() override;

private:
    void scannerLoop();
    void discoverClient(int slot);
    void cleanupDisconnected();
    std::string getRegistryName();

    std::atomic<bool> running;
    ClientRegistry* registry;
    std::thread scannerThread;
    std::function<void(std::unique_ptr<IChannel>)> callback;

    // 记录正在服务的 slot，防止重复创建
    std::mutex internalMutex;
    std::vector<int> activeSlots; 
};