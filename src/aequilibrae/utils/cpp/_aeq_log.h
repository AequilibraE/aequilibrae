#ifndef HELPER_AEQ_LOG_H
#define HELPER_AEQ_LOG_H

// Includes required by bridge.h
#include <string>
#include <sstream>

#define AEQ_LOG(bridge, level, msg_exp)                                        \
  do {                                                                         \
    if ((level) >= (bridge)->c_level) {                                        \
      (bridge)->log_wrapper_func((bridge), (level), (msg_exp));                \
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
