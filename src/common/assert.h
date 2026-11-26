#ifndef KYTY_COMMON_ASSERT_H_
#define KYTY_COMMON_ASSERT_H_

#include "common/common.h"
#include "common/logging/log.h"

#include <string_view>

namespace Common {

[[noreturn]] void DbgExitHandler(char const* file, int line, std::string_view text);
[[noreturn]] void DbgExitIfHandler(char const* expr, char const* file, int line);
[[noreturn]] void DbgNotImplementedHandler(char const* expr, char const* file, int line);

// Keep-going support (common/recoverable.h). While a hook is set on the calling thread, a failed
// EXIT/EXIT_IF/EXIT_NOT_IMPLEMENTED calls it instead of terminating. The hook must not return.
using RecoverHook = void (*)(std::string_view text, char const* file, int line);
RecoverHook SetThreadRecoverHook(RecoverHook hook);

} // namespace Common

#ifndef KYTY_FINAL
#define EXIT_IF(x) ((x) ? Common::DbgExitIfHandler(#x, __FILE__, __LINE__) : (void)0)
#else
#define EXIT_IF(x) ((void)(false && (x)))
#endif
#define EXIT(...) Common::DbgExitHandler(__FILE__, __LINE__, ::fmt::sprintf(__VA_ARGS__))
#define EXIT_NOT_IMPLEMENTED(x)                                                                    \
	((x) ? Common::DbgNotImplementedHandler(#x, __FILE__, __LINE__) : (void)0)

#endif /* KYTY_COMMON_ASSERT_H_ */
