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

// ============================================================
//  常量定义
// ============================================================

#define SCHEDULER_PORT 9999        // 保留用于兼容性（已不再使用）
#define LOCALHOST "127.0.0.1"      // 保留用于兼容性（已不再使用）

// SPSC 队列配置
constexpr size_t SPSC_QUEUE_SIZE = 1024;        // 队列可存储的消息数量
constexpr size_t SPSC_MSG_SIZE = 256;           // 每条消息的最大字节数
constexpr size_t CACHE_LINE_SIZE = 64;          // CPU 缓存行大小，用于避免伪共享

// 共享内存名称
#define SHM_NAME_PYTORCH "/kernel_scheduler_pytorch"
#define SHM_NAME_SGLANG  "/kernel_scheduler_sglang"

// ============================================================
//  消息构建函数（保持兼容）
// ============================================================

static inline std::string createRequestMessage(const std::string& id, const std::string& type) {
    return type + "|" + id + "|pytorch\n";
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
};
