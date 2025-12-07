#pragma once

#include <string>
#include <cstdint>
#include <cstring>
#include <atomic>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <string_view>

// ============================================================
//  常量定义
// ============================================================

#define SCHEDULER_PORT 9999        // 保留用于兼容性（已不再使用）
#define LOCALHOST "127.0.0.1"      // 保留用于兼容性（已不再使用）

// SPSC 队列配置
constexpr size_t SPSC_QUEUE_SIZE = 1024;        // 队列可存储的消息数量
constexpr size_t SPSC_MSG_SIZE = 256;           // 每条消息的最大字节数
constexpr size_t CACHE_LINE_SIZE = 64;          // CPU 缓存行大小，用于避免伪共享

// 共享内存名称前缀（支持多客户端动态通道）
#define SHM_NAME_PREFIX_PYTORCH "/ks_pytorch_"
#define SHM_NAME_PREFIX_SGLANG  "/ks_sglang_"

// 保留旧名称用于兼容（单客户端模式）
#define SHM_NAME_PYTORCH "/kernel_scheduler_pytorch"
#define SHM_NAME_SGLANG  "/kernel_scheduler_sglang"

// 注册通道（客户端在此注册自己的通道名）
constexpr size_t MAX_REGISTERED_CLIENTS = 64;  // 最多支持的客户端数量

// ============================================================
//  用户名后缀，避免不同用户的共享内存冲突
// ============================================================
inline std::string get_user_suffix() {
    const char* u = std::getenv("USER");
    if (u && *u) {
        return std::string("_") + u;
    }
    return "_nouser";
}

inline std::string get_registry_name() {
    return std::string("/kernel_scheduler_registry") + get_user_suffix();
}

// ============================================================
//  消息构建函数（保持兼容）
// ============================================================

static inline std::string createRequestMessage(const std::string& id, const std::string& type, const std::string& unique_id = "") {
    if (unique_id.empty()) {
        return type + "|" + id + "|pytorch\n";
    }
    return type + "|" + id + "|pytorch|" + unique_id + "\n";
}

static inline std::string createResponseMessage(const std::string& id, bool allowed, const std::string& reason) {
    return id + "|" + (allowed ? "1" : "0") + "|" + reason + "\n";
}

// ============================================================
//  SPSC 无锁环形队列
// ============================================================

/**
 * SPSCQueue - 单生产者单消费者无锁队列
 * 
 * 设计要点：
 * 1. head 和 tail 分别在不同缓存行，避免伪共享
 * 2. 生产者只修改 tail，消费者只修改 head
 * 3. 使用 acquire-release 语义保证内存顺序
 */
struct SPSCQueue {
    // 对齐到缓存行，避免伪共享
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> head;  // 消费者读取位置
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> tail;  // 生产者写入位置
    
    // 消息缓冲区
    alignas(CACHE_LINE_SIZE) char buffer[SPSC_QUEUE_SIZE][SPSC_MSG_SIZE];

    // 初始化队列
    void init() {
        head.store(0, std::memory_order_relaxed);
        tail.store(0, std::memory_order_relaxed);
        std::memset(buffer, 0, sizeof(buffer));
    }

    // 生产者：尝试写入消息
    // 返回 true 表示成功，false 表示队列已满
    bool try_push(const char* data, size_t len) {
        uint64_t current_tail = tail.load(std::memory_order_relaxed);
        uint64_t next_tail = (current_tail + 1) % SPSC_QUEUE_SIZE;
        
        // 检查队列是否已满
        if (next_tail == head.load(std::memory_order_acquire)) {
            return false;  // 队列已满
        }

        // 写入数据
        size_t copy_len = (len < SPSC_MSG_SIZE - 1) ? len : (SPSC_MSG_SIZE - 1);
        std::memcpy(buffer[current_tail], data, copy_len);
        buffer[current_tail][copy_len] = '\0';

        // 发布写入（使用 release 语义确保数据对消费者可见）
        tail.store(next_tail, std::memory_order_release);
        return true;
    }

    // 便捷版本：直接接受 std::string
    bool try_push(const std::string& msg) {
        return try_push(msg.c_str(), msg.length());
    }

    // 消费者：尝试读取消息
    // 返回 true 表示成功，false 表示队列为空
    bool try_pop(char* out_data, size_t max_len) {
        uint64_t current_head = head.load(std::memory_order_relaxed);
        
        // 检查队列是否为空
        if (current_head == tail.load(std::memory_order_acquire)) {
            return false;  // 队列为空
        }

        // 读取数据
        size_t copy_len = std::strlen(buffer[current_head]);
        if (copy_len >= max_len) copy_len = max_len - 1;
        std::memcpy(out_data, buffer[current_head], copy_len);
        out_data[copy_len] = '\0';

        // 提交读取（使用 release 语义确保读取完成后再更新 head）
        head.store((current_head + 1) % SPSC_QUEUE_SIZE, std::memory_order_release);
        return true;
    }

    // 便捷版本：返回 std::string，如果队列为空则返回空字符串
    std::string try_pop() {
        char buf[SPSC_MSG_SIZE];
        if (try_pop(buf, SPSC_MSG_SIZE)) {
            return std::string(buf);
        }
        return "";
    }

    // 阻塞式读取（带超时）- 高性能版本：纯忙等待
    bool pop_blocking(char* out_data, size_t max_len, int timeout_ms = -1) {
        if (timeout_ms < 0) {
            // 无超时，纯忙等待（最高性能）
            while (!try_pop(out_data, max_len)) {
                __asm__ __volatile__("pause" ::: "memory");
            }
            return true;
        } else {
            // 有超时
            auto start = std::chrono::steady_clock::now();
            while (true) {
                if (try_pop(out_data, max_len)) {
                    return true;
                }
                auto now = std::chrono::steady_clock::now();
                auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
                if (elapsed_ms >= timeout_ms) {
                    return false;
                }
                __asm__ __volatile__("pause" ::: "memory");
            }
        }
    }

    // 阻塞式写入（带超时）- 高性能版本：纯忙等待
    bool push_blocking(const char* data, size_t len, int timeout_ms = -1) {
        if (timeout_ms < 0) {
            // 无超时，纯忙等待（最高性能）
            while (!try_push(data, len)) {
                __asm__ __volatile__("pause" ::: "memory");
            }
            return true;
        } else {
            // 有超时
            auto start = std::chrono::steady_clock::now();
            while (true) {
                if (try_push(data, len)) {
                    return true;
                }
                auto now = std::chrono::steady_clock::now();
                auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
                if (elapsed_ms >= timeout_ms) {
                    return false;
                }
                __asm__ __volatile__("pause" ::: "memory");
            }
        }
    }

    bool push_blocking(const std::string& msg, int timeout_ms = -1) {
        return push_blocking(msg.c_str(), msg.length(), timeout_ms);
    }

    // 检查队列是否为空
    bool empty() const {
        return head.load(std::memory_order_acquire) == tail.load(std::memory_order_acquire);
    }

    // 获取队列中消息数量
    size_t size() const {
        uint64_t h = head.load(std::memory_order_acquire);
        uint64_t t = tail.load(std::memory_order_acquire);
        return (t - h + SPSC_QUEUE_SIZE) % SPSC_QUEUE_SIZE;
    }
};

// ============================================================
//  客户端通道 - 包含双向 SPSC 队列
// ============================================================

/**
 * ClientChannel - 客户端与调度器之间的双向通信通道
 * 
 * 每个客户端（pytorch/sglang）有独立的通道
 */
struct ClientChannel {
    SPSCQueue request_queue;    // 客户端 -> 调度器
    SPSCQueue response_queue;   // 调度器 -> 客户端
    
    // 连接标志
    alignas(CACHE_LINE_SIZE) std::atomic<bool> client_connected;
    alignas(CACHE_LINE_SIZE) std::atomic<bool> scheduler_ready;
    
    // 初始化通道
    void init() {
        request_queue.init();
        response_queue.init();
        client_connected.store(false, std::memory_order_relaxed);
        scheduler_ready.store(false, std::memory_order_relaxed);
    }
};

// ============================================================
//  客户端注册表 - 用于动态多客户端支持
// ============================================================

/**
 * ClientRegistryEntry - 单个客户端的注册信息
 */
struct ClientRegistryEntry {
    alignas(CACHE_LINE_SIZE) std::atomic<bool> active;           // 是否活跃
    char shm_name[64];                                            // 共享内存名称
    char client_type[16];                                         // 客户端类型：pytorch/sglang
    char unique_id[64];                                           // 客户端唯一标识（UNIQUE_ID 环境变量）
    alignas(CACHE_LINE_SIZE) std::atomic<int64_t> client_pid;    // 客户端进程 PID（用于检测进程存活）
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> last_heartbeat; // 最后心跳时间戳
    
    void init() {
        active.store(false, std::memory_order_relaxed);
        std::memset(shm_name, 0, sizeof(shm_name));
        std::memset(client_type, 0, sizeof(client_type));
        std::memset(unique_id, 0, sizeof(unique_id));
        client_pid.store(0, std::memory_order_relaxed);
        last_heartbeat.store(0, std::memory_order_relaxed);
    }
};

/**
 * ClientRegistry - 客户端注册表
 * 
 * 客户端启动时在此注册自己的通道名，调度器扫描此表发现新客户端
 */
struct ClientRegistry {
    alignas(CACHE_LINE_SIZE) std::atomic<bool> scheduler_ready;  // 调度器是否已准备好
    alignas(CACHE_LINE_SIZE) std::atomic<uint32_t> version;      // 版本号，每次有变更时递增
    ClientRegistryEntry entries[MAX_REGISTERED_CLIENTS];
    
    void init() {
        scheduler_ready.store(false, std::memory_order_relaxed);
        version.store(0, std::memory_order_relaxed);
        for (size_t i = 0; i < MAX_REGISTERED_CLIENTS; i++) {
            entries[i].init();
        }
    }
    
    // 注册新客户端，返回分配的槽位索引，失败返回 -1
    int register_client(const char* shm_name, const char* client_type, const char* unique_id, int64_t pid = 0) {
        for (size_t i = 0; i < MAX_REGISTERED_CLIENTS; i++) {
            bool expected = false;
            if (entries[i].active.compare_exchange_strong(expected, true, 
                    std::memory_order_acq_rel, std::memory_order_relaxed)) {
                // 成功占用此槽位
                std::strncpy(entries[i].shm_name, shm_name, sizeof(entries[i].shm_name) - 1);
                std::strncpy(entries[i].client_type, client_type, sizeof(entries[i].client_type) - 1);
                std::strncpy(entries[i].unique_id, unique_id, sizeof(entries[i].unique_id) - 1);
                entries[i].client_pid.store(pid, std::memory_order_release);
                entries[i].last_heartbeat.store(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()
                    ).count(), std::memory_order_release);
                version.fetch_add(1, std::memory_order_release);
                return static_cast<int>(i);
            }
        }
        return -1;  // 没有空闲槽位
    }
    
    // 注销客户端
    void unregister_client(int slot) {
        if (slot >= 0 && slot < static_cast<int>(MAX_REGISTERED_CLIENTS)) {
            entries[slot].active.store(false, std::memory_order_release);
            version.fetch_add(1, std::memory_order_release);
        }
    }
    
    // 更新心跳
    void update_heartbeat(int slot) {
        if (slot >= 0 && slot < static_cast<int>(MAX_REGISTERED_CLIENTS)) {
            entries[slot].last_heartbeat.store(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()
                ).count(), std::memory_order_release);
        }
    }
};

// ============================================================
//  共享内存辅助类
// ============================================================

/**
 * SharedMemoryHelper - 管理共享内存的创建和映射
 */
class SharedMemoryHelper {
public:
    // 创建或打开共享内存（服务端/调度器使用）
    static ClientChannel* create_or_open(const char* shm_name, bool create = false) {
        int flags = O_RDWR;
        if (create) {
            flags |= O_CREAT;
        }
        
        int fd = shm_open(shm_name, flags, 0666);
        if (fd == -1) {
            std::cerr << "[SHM] 打开共享内存失败: " << shm_name << std::endl;
            return nullptr;
        }

        if (create) {
            // 设置共享内存大小
            if (ftruncate(fd, sizeof(ClientChannel)) == -1) {
                std::cerr << "[SHM] 设置共享内存大小失败" << std::endl;
                close(fd);
                return nullptr;
            }
        }

        // 映射到进程地址空间
        void* ptr = mmap(nullptr, sizeof(ClientChannel), 
                        PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);  // 映射后可以关闭 fd

        if (ptr == MAP_FAILED) {
            std::cerr << "[SHM] 映射共享内存失败" << std::endl;
            return nullptr;
        }

        ClientChannel* channel = static_cast<ClientChannel*>(ptr);
        
        if (create) {
            channel->init();
        }

        return channel;
    }

    // 解除映射
    static void unmap(ClientChannel* channel) {
        if (channel) {
            munmap(channel, sizeof(ClientChannel));
        }
    }

    // 删除共享内存（仅调度器在退出时使用）
    static void unlink(const char* shm_name) {
        shm_unlink(shm_name);
    }
    
    // 创建或打开注册表共享内存
    static ClientRegistry* create_or_open_registry(bool create = false) {
        int flags = O_RDWR;
        if (create) {
            flags |= O_CREAT;
        }
        
        std::string reg_name = get_registry_name();
        int fd = shm_open(reg_name.c_str(), flags, 0666);
        if (fd == -1) {
            std::cerr << "[SHM] 打开注册表共享内存失败: " << reg_name << std::endl;
            return nullptr;
        }

        if (create) {
            if (ftruncate(fd, sizeof(ClientRegistry)) == -1) {
                std::cerr << "[SHM] 设置注册表共享内存大小失败" << std::endl;
                close(fd);
                return nullptr;
            }
        }

        void* ptr = mmap(nullptr, sizeof(ClientRegistry), 
                        PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);

        if (ptr == MAP_FAILED) {
            std::cerr << "[SHM] 映射注册表共享内存失败" << std::endl;
            return nullptr;
        }

        ClientRegistry* registry = static_cast<ClientRegistry*>(ptr);
        
        if (create) {
            registry->init();
        }

        return registry;
    }
    
    // 解除注册表映射
    static void unmap_registry(ClientRegistry* registry) {
        if (registry) {
            munmap(registry, sizeof(ClientRegistry));
        }
    }
    
    // 删除注册表共享内存
    static void unlink_registry() {
        std::string reg_name = get_registry_name();
        shm_unlink(reg_name.c_str());
    }
};
