#include "common/faultReport.h"

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <vector>

#include <dlfcn.h>
#include <elf.h>
#include <execinfo.h>
#include <fcntl.h>
#include <limits.h>
#include <link.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>

namespace Common::FaultReport {

namespace {

constexpr size_t MAX_MODULES = 512;
constexpr size_t NAME_SIZE   = 256;

struct Module {
	uint64_t start = 0; // the mapped range of one loadable segment
	uint64_t end   = 0;
	uint64_t base  = 0; // load bias: address - base is the offset in the shared object
	char     name[NAME_SIZE] {};
};

// Published without a lock: the count is stored last, so a handler that runs
// during a refresh sees a prefix of the table. Entries are only appended or rewritten in place
// with identical content for the same module.
Module                g_modules[MAX_MODULES];
std::atomic<size_t>   g_module_count {0};
std::atomic<bool>     g_refreshing {false};

struct Symbol {
	uint64_t    address;
	uint64_t    size;
	const char* name;
};
std::vector<Symbol> g_symbols; // functions of the main executable, sorted by address
uint64_t            g_exe_base = 0;
bool                g_exe_known = false;

struct StackLimits {
	uintptr_t low  = 0;
	uintptr_t high = 0;
};
thread_local StackLimits t_stack;
thread_local const char* t_tag = nullptr;

void LoadExecutableSymbols(const char* path) {
	const int fd = ::open(path, O_RDONLY | O_CLOEXEC);
	if (fd < 0) {
		return;
	}
	struct stat st {};
	if (::fstat(fd, &st) != 0 || st.st_size < static_cast<off_t>(sizeof(Elf64_Ehdr))) {
		::close(fd);
		return;
	}
	void* map = ::mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
	::close(fd);
	if (map == MAP_FAILED) {
		return;
	}
	// The mapping stays: symbol names point into it, and untouched pages cost no memory.
	const auto* bytes = static_cast<const uint8_t*>(map);
	const auto  size  = static_cast<uint64_t>(st.st_size);
	const auto* eh    = reinterpret_cast<const Elf64_Ehdr*>(bytes);
	if (std::memcmp(eh->e_ident, ELFMAG, SELFMAG) != 0 || eh->e_ident[EI_CLASS] != ELFCLASS64 ||
	    eh->e_shoff == 0 || eh->e_shoff + static_cast<uint64_t>(eh->e_shnum) * sizeof(Elf64_Shdr) > size) {
		return;
	}
	const auto* sh = reinterpret_cast<const Elf64_Shdr*>(bytes + eh->e_shoff);
	for (unsigned i = 0; i < eh->e_shnum; i++) {
		if (sh[i].sh_type != SHT_SYMTAB && sh[i].sh_type != SHT_DYNSYM) {
			continue;
		}
		if (sh[i].sh_link >= eh->e_shnum || sh[i].sh_offset + sh[i].sh_size > size) {
			continue;
		}
		const auto& str = sh[sh[i].sh_link];
		if (str.sh_offset + str.sh_size > size) {
			continue;
		}
		const auto* syms  = reinterpret_cast<const Elf64_Sym*>(bytes + sh[i].sh_offset);
		const auto  count = sh[i].sh_size / sizeof(Elf64_Sym);
		const auto* names = reinterpret_cast<const char*>(bytes + str.sh_offset);
		for (uint64_t s = 0; s < count; s++) {
			if (ELF64_ST_TYPE(syms[s].st_info) != STT_FUNC || syms[s].st_value == 0 ||
			    syms[s].st_name >= str.sh_size) {
				continue;
			}
			g_symbols.push_back({syms[s].st_value, syms[s].st_size, names + syms[s].st_name});
		}
	}
	std::sort(g_symbols.begin(), g_symbols.end(),
	          [](const Symbol& a, const Symbol& b) { return a.address < b.address; });
}

// Looks up an exe function by link-time address (address - load bias).
const Symbol* FindExeSymbol(uint64_t link_address) {
	if (g_symbols.empty()) {
		return nullptr;
	}
	auto it = std::upper_bound(g_symbols.begin(), g_symbols.end(), link_address,
	                           [](uint64_t a, const Symbol& s) { return a < s.address; });
	while (it != g_symbols.begin()) {
		--it;
		if (link_address < it->address + std::max<uint64_t>(it->size, 1)) {
			return &*it;
		}
		if (it->size != 0) {
			break; // sized symbols do not nest here; stop at the nearest sized one
		}
	}
	return nullptr;
}

bool FindModule(uint64_t address, const Module** out) {
	const auto count = std::min(g_module_count.load(std::memory_order_acquire), MAX_MODULES);
	for (size_t i = 0; i < count; i++) {
		if (address >= g_modules[i].start && address < g_modules[i].end) {
			*out = &g_modules[i];
			return true;
		}
	}
	return false;
}

// Fallback for addresses of modules loaded after the last refresh (the Vulkan loader loads its
// drivers itself): reads /proc/self/maps with read(2) only, line by line. The offset base is the
// lowest mapping of the same file.
bool FindInMaps(uint64_t address, Module* out) {
	const int fd = ::open("/proc/self/maps", O_RDONLY | O_CLOEXEC);
	if (fd < 0) {
		return false;
	}
	char     buffer[4096];
	char     line[NAME_SIZE + 128];
	size_t   line_length = 0;
	bool     found       = false;
	uint64_t found_start = 0;
	uint64_t found_end   = 0;
	char     found_name[NAME_SIZE] {};
	uint64_t lowest = 0;
	// Pass over all lines: remember the matching line, then the lowest start for its file name.
	auto handle = [&](const char* l) {
		uint64_t start = 0;
		uint64_t end   = 0;
		int      consumed = 0;
		if (std::sscanf(l, "%" SCNx64 "-%" SCNx64 " %*s %*s %*s %*s %n", &start, &end, &consumed) < 2 ||
		    consumed == 0) {
			return;
		}
		const char* path = l + consumed;
		if (!found && address >= start && address < end) {
			found       = true;
			found_start = start;
			found_end   = end;
			std::snprintf(found_name, sizeof(found_name), "%s", path);
			lowest = start;
		} else if (found && std::strcmp(path, found_name) == 0 && start < lowest) {
			lowest = start;
		}
	};
	for (;;) {
		const auto n = ::read(fd, buffer, sizeof(buffer));
		if (n <= 0) {
			break;
		}
		for (ssize_t i = 0; i < n; i++) {
			if (buffer[i] == '\n') {
				line[line_length] = '\0';
				handle(line);
				line_length = 0;
			} else if (line_length + 1 < sizeof(line)) {
				line[line_length++] = buffer[i];
			}
		}
	}
	::close(fd);
	if (!found) {
		return false;
	}
	// Mappings before the match of the same file were seen as "not found yet": a second pass
	// finds the lowest start of the file (the load base of a shared object).
	const int fd2 = ::open("/proc/self/maps", O_RDONLY | O_CLOEXEC);
	if (fd2 >= 0) {
		line_length = 0;
		for (;;) {
			const auto n = ::read(fd2, buffer, sizeof(buffer));
			if (n <= 0) {
				break;
			}
			for (ssize_t i = 0; i < n; i++) {
				if (buffer[i] == '\n') {
					line[line_length] = '\0';
					uint64_t start = 0;
					uint64_t end   = 0;
					int      consumed = 0;
					if (std::sscanf(line, "%" SCNx64 "-%" SCNx64 " %*s %*s %*s %*s %n", &start, &end,
					                &consumed) >= 2 &&
					    consumed != 0 && std::strcmp(line + consumed, found_name) == 0 &&
					    start < lowest) {
						lowest = start;
					}
					line_length = 0;
				} else if (line_length + 1 < sizeof(line)) {
					line[line_length++] = buffer[i];
				}
			}
		}
		::close(fd2);
	}
	out->start = found_start;
	out->end   = found_end;
	out->base  = lowest;
	std::snprintf(out->name, sizeof(out->name), "%s", found_name[0] != '\0' ? found_name : "[anonymous]");
	return true;
}

const char* BaseName(const char* path) {
	const char* slash = std::strrchr(path, '/');
	return slash != nullptr ? slash + 1 : path;
}

bool IsReadable(uint64_t address, size_t size) {
	if (address == 0 || address + size < address) {
		return false;
	}
	uint8_t probe = 0;
	iovec   local {&probe, 1};
	iovec   remote {reinterpret_cast<void*>(address), 1};
	if (::process_vm_readv(::getpid(), &local, 1, &remote, 1, 0) != 1) {
		return false;
	}
	if (size > 1) {
		remote.iov_base = reinterpret_cast<void*>(address + size - 1);
		return ::process_vm_readv(::getpid(), &local, 1, &remote, 1, 0) == 1;
	}
	return true;
}

int Callback(dl_phdr_info* info, size_t, void* user) {
	auto*      exe_name = static_cast<const char*>(user);
	const char* name    = info->dlpi_name != nullptr && info->dlpi_name[0] != '\0' ? info->dlpi_name : exe_name;
	for (unsigned i = 0; i < info->dlpi_phnum; i++) {
		const auto& ph = info->dlpi_phdr[i];
		if (ph.p_type != PT_LOAD) {
			continue;
		}
		const auto index = g_module_count.load(std::memory_order_relaxed);
		if (index >= MAX_MODULES) {
			return 1;
		}
		auto& m = g_modules[index];
		m.start = info->dlpi_addr + ph.p_vaddr;
		m.end   = m.start + ph.p_memsz;
		m.base  = info->dlpi_addr;
		std::snprintf(m.name, sizeof(m.name), "%s", name);
		g_module_count.store(index + 1, std::memory_order_release);
	}
	return 0;
}

void Out(const Sink& sink, const char* fmt, ...) __attribute__((format(printf, 2, 3)));
void Out(const Sink& sink, const char* fmt, ...) {
	char    text[768];
	va_list args;
	va_start(args, fmt);
	std::vsnprintf(text, sizeof(text), fmt, args);
	va_end(args);
	sink.write(sink.user, text);
}

// Reentrancy guard for dladdr: the dynamic loader's lock must not be taken by a thread that faulted
// inside the loader itself.
bool InLoader(uint64_t address) {
	const Module* m = nullptr;
	if (!FindModule(address, &m)) {
		return false;
	}
	const char* base = BaseName(m->name);
	return std::strncmp(base, "ld-linux", 8) == 0 || std::strncmp(base, "ld-", 3) == 0;
}

} // namespace

void SetThreadTag(const char* tag, const char* name) noexcept {
	t_tag = tag;
	if (name != nullptr) {
		::pthread_setname_np(::pthread_self(), name);
	}
}

void RefreshModules() {
	if (g_refreshing.exchange(true)) {
		return;
	}
	char exe[NAME_SIZE] {};
	const auto n = ::readlink("/proc/self/exe", exe, sizeof(exe) - 1);
	if (n > 0) {
		exe[n] = '\0';
	}
	g_module_count.store(0, std::memory_order_release);
	::dl_iterate_phdr(Callback, exe);
	if (g_symbols.empty() && exe[0] != '\0') {
		// The first module is the main executable.
		if (g_module_count.load() > 0) {
			g_exe_base  = g_modules[0].base;
			g_exe_known = true;
		}
		LoadExecutableSymbols(exe);
	}
	// Prime the unwinder so the handler's first backtrace() does not allocate or load libgcc.
	void* warm[4];
	(void)::backtrace(warm, 4);
	g_refreshing.store(false);
}

void RegisterThreadStack() {
	pthread_attr_t attr;
	if (::pthread_getattr_np(::pthread_self(), &attr) != 0) {
		return;
	}
	void*  addr = nullptr;
	size_t size = 0;
	if (::pthread_attr_getstack(&attr, &addr, &size) == 0) {
		t_stack.low  = reinterpret_cast<uintptr_t>(addr);
		t_stack.high = t_stack.low + size;
	}
	::pthread_attr_destroy(&attr);
}

void FormatAddress(uint64_t address, char* out, size_t size) {
	const Module* module = nullptr;
	Module        from_maps;
	if (!FindModule(address, &module)) {
		if (FindInMaps(address, &from_maps)) {
			module = &from_maps;
		}
	}
	if (module == nullptr) {
		std::snprintf(out, size, "not in a loaded module");
		return;
	}
	const uint64_t offset = address - module->base;
	const char*    symbol = nullptr;
	uint64_t       sym_offset = 0;
	if (const auto* s = (g_exe_known && module->base == g_exe_base)
	                        ? FindExeSymbol(offset)
	                        : nullptr;
	    s != nullptr) {
		symbol     = s->name;
		sym_offset = offset - s->address;
	} else if (!InLoader(address)) {
		Dl_info dl {};
		if (::dladdr(reinterpret_cast<void*>(address), &dl) != 0 && dl.dli_sname != nullptr) {
			symbol     = dl.dli_sname;
			sym_offset = address - reinterpret_cast<uint64_t>(dl.dli_saddr);
		}
	}
	if (symbol != nullptr) {
		std::snprintf(out, size, "%s+0x%" PRIx64 " [%s+0x%" PRIx64 "]", BaseName(module->name), offset,
		              symbol, sym_offset);
	} else {
		std::snprintf(out, size, "%s+0x%" PRIx64, BaseName(module->name), offset);
	}
}

void WriteHostContext(const Registers& regs, const Sink& sink) {
	char name[32] {};
	::pthread_getname_np(::pthread_self(), name, sizeof(name));
	Out(sink, "--- Host fault context ---\n");
	Out(sink, "host thread: %s tid=%ld doing: %s\n", name[0] != '\0' ? name : "(unnamed)",
	    static_cast<long>(::gettid()), t_tag != nullptr ? t_tag : "(no tag)");
	char where[NAME_SIZE + 160];
	FormatAddress(regs.pc, where, sizeof(where));
	Out(sink, " pc       %016" PRIx64 " = %s\n", regs.pc, where);

	// Frame-pointer chain bounded by the thread's stack, or 1 MiB above the stack pointer.
	uintptr_t low  = regs.rsp;
	uintptr_t high = regs.rsp + 1024u * 1024u;
	if (t_stack.low <= regs.rsp && regs.rsp < t_stack.high) {
		high = t_stack.high;
	}
	Out(sink, "frame-pointer chain (code built without frame pointers may skip frames):\n");
	uint64_t frame = regs.rbp;
	int      depth = 0;
	while (depth < 24 && frame >= low && frame + 16 <= high && (frame & 7u) == 0 &&
	       IsReadable(frame, 16)) {
		const auto* words = reinterpret_cast<const uint64_t*>(frame);
		FormatAddress(words[1], where, sizeof(where));
		Out(sink, " fp[%2d]   %016" PRIx64 " = %s\n", depth, words[1], where);
		if (words[0] <= frame) {
			break;
		}
		frame = words[0];
		depth++;
	}
	if (depth == 0) {
		Out(sink, " (none)\n");
	}

	// The unwinder reads .eh_frame, so it also follows frames without a frame pointer; its first
	// frames are the handler and the signal trampoline.
	void*      frames[48];
	const auto count = ::backtrace(frames, 48);
	Out(sink, "backtrace (unwinder, %d frames, handler frames first):\n", count);
	for (int i = 0; i < count; i++) {
		FormatAddress(reinterpret_cast<uint64_t>(frames[i]), where, sizeof(where));
		Out(sink, " bt[%2d]   %016" PRIx64 " = %s\n", i, reinterpret_cast<uint64_t>(frames[i]), where);
	}

	// Stack words that look like return addresses (executable module ranges only).
	Out(sink, "stack words that point into modules (rsp upward):\n");
	int named = 0;
	for (int i = 0; i < 128 && named < 24; i++) {
		const uint64_t addr = regs.rsp + static_cast<uint64_t>(i) * 8u;
		if (addr + 8 > high || !IsReadable(addr, 8)) {
			break;
		}
		const uint64_t value = *reinterpret_cast<const uint64_t*>(addr);
		const Module*  m     = nullptr;
		if (value > 0x10000 && FindModule(value, &m)) {
			FormatAddress(value, where, sizeof(where));
			Out(sink, " rsp+0x%03x %016" PRIx64 " = %s\n", i * 8, value, where);
			named++;
		}
	}
}

} // namespace Common::FaultReport

#else // not Linux: only the tag API is meaningful

namespace Common::FaultReport {
void SetThreadTag(const char*, const char*) noexcept {}
void RefreshModules() {}
void RegisterThreadStack() {}
void FormatAddress(uint64_t, char* out, size_t size) {
	if (size > 0) {
		out[0] = '\0';
	}
}
void WriteHostContext(const Registers&, const Sink&) {}
} // namespace Common::FaultReport

#endif
