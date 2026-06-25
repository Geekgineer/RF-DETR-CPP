// SPDX-FileCopyrightText: 2024 Geekgineer
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

// ============================================================================
// RFDETRs-CPP — Bounded thread-safe queue for camera/video pipelines
// ============================================================================

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>

template <typename T>
class BoundedThreadSafeQueue {
public:
    explicit BoundedThreadSafeQueue(size_t maxSize) : maxSize_(maxSize), finished_(false) {}

    bool enqueue(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cvNotFull_.wait(lock, [this] { return queue_.size() < maxSize_ || finished_; });
        if (finished_) return false;
        queue_.push(std::move(item));
        cvNotEmpty_.notify_one();
        return true;
    }

    bool dequeue(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cvNotEmpty_.wait(lock, [this] { return !queue_.empty() || finished_; });
        if (queue_.empty()) return false;
        item = std::move(queue_.front());
        queue_.pop();
        cvNotFull_.notify_one();
        return true;
    }

    void set_finished() {
        std::unique_lock<std::mutex> lock(mutex_);
        finished_ = true;
        cvNotEmpty_.notify_all();
        cvNotFull_.notify_all();
    }

private:
    std::queue<T> queue_;
    size_t maxSize_;
    bool finished_;
    std::mutex mutex_;
    std::condition_variable cvNotEmpty_;
    std::condition_variable cvNotFull_;
};
