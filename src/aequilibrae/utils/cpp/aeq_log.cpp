#include "aeq_log.h"

void cpp_function_that_logs(struct Bridge *b) {
  std::string s1("hello from cpp");
  AEQ_LOG(b, AEQ_LOG_CRITICAL, s1);
}
