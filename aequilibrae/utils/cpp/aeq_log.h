#ifndef AEQ_LOG_H
#define AEQ_LOG_H

// Includes required by bridge.h
#include <atomic>
#include <mutex>
#include <deque>
#include <string>
#include <sstream>
// Can't include here because duplicate declarations?
// #include "bridge.h"  // This header is generated when bridge.pyx is cythonized

#define AEQ_LOG(bridge, level, msg_exp)                                        \
  do {                                                                         \
    if ((level) >= (bridge)->c_level) {                                        \
      aeq_c_to_python_log_bridge((bridge), (level), (msg_exp));                \
    }                                                                          \
  } while (0)

// std::format in C++20 is pretty nice but this will do for now
template <typename... Args>
std::string aeq_format_string(Args &&...args) {
  std::ostringstream oss;
  (oss << ... << args);
  return oss.str();
}

#endif  // AEQ_LOG_H
