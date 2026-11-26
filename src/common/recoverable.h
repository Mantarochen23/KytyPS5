#ifndef KYTY_COMMON_RECOVERABLE_H_
#define KYTY_COMMON_RECOVERABLE_H_

// Keep-going mode: turns a failed EXIT/EXIT_IF/EXIT_NOT_IMPLEMENTED on this thread into a C++
// exception for the lifetime of a RecoverableScope. Include only from translation units built with
// exceptions enabled, and only around code whose partial effects the catch site can discard.

#include "common/assert.h"

#include <stdexcept>
#include <string>

namespace Common {

struct RecoverableFailure: std::runtime_error {
	RecoverableFailure(std::string_view text, const char* file_name, int line_number)
	    : std::runtime_error(std::string(text)), file(file_name), line(line_number) {}
	const char* file;
	int         line;
};

class RecoverableScope {
public:
	RecoverableScope(): m_previous(SetThreadRecoverHook(&Throw)) {}
	~RecoverableScope() { SetThreadRecoverHook(m_previous); }
	RecoverableScope(const RecoverableScope&)            = delete;
	RecoverableScope& operator=(const RecoverableScope&) = delete;

private:
	[[noreturn]] static void Throw(std::string_view text, const char* file, int line) {
		throw RecoverableFailure(text, file, line);
	}
	RecoverHook m_previous;
};

} // namespace Common

#endif /* KYTY_COMMON_RECOVERABLE_H_ */
