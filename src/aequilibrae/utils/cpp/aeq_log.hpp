#pragma once

#include <deque>
#include <mutex>
#include <sstream>
#include <string>

class AeqLogClosure {
public:
  std::mutex _log_queue_mutex;
  std::deque<std::pair<int, std::string>> _log_queue;
  uint8_t c_level;

  inline void _log(uint8_t level, std::string msg) {
    std::unique_lock<std::mutex> lock{this->_log_queue_mutex};
    this->_log_queue.emplace_back(level, msg);
  }
};

#define AEQ_LOG_DEBUG 10
#define AEQ_LOG_INFO 20
#define AEQ_LOG_WARNING 30
#define AEQ_LOG_ERROR 40
#define AEQ_LOG_CRITICAL 50

#define AEQ_LOG(closure, level, msg_exp)                                       \
  do {                                                                         \
    if ((closure) && (level) >= (closure)->c_level) {                          \
      (closure)->_log((level), (msg_exp));                                     \
    }                                                                          \
  } while (0)

// std::format in C++20 is pretty nice but this will do for now
template <typename... Args> std::string aeq_format_string(Args &&...args) {
  std::ostringstream oss;
  (oss << ... << args);
  return oss.str();
}
