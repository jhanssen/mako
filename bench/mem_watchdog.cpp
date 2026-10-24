// Runs a command in its own process group and kills the whole group when the machine runs short of memory:
// available memory (free + purgeable + clean file cache) below a floor, or the compressor holding more than a
// limit. Logs memory every second (and the kill) to a file, fsync'ed per line so it survives a forced restart.
//   mem_watchdog --log FILE [--min-avail-gb G] [--max-compressed-gb G] -- command [args...]
#include <mach/mach.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Mem {
    double free = 0, avail = 0, compressed = 0, wired = 0, swapouts = 0;
};

Mem read_mem()
{
    vm_statistics64_data_t vs{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vs), &count);
    const double page = double(vm_page_size);
    Mem m;
    m.free = double(vs.free_count) * page;
    const double clean = std::min<double>(double(vs.inactive_count) + vs.speculative_count, vs.external_page_count);
    m.avail = (double(vs.free_count) + vs.purgeable_count + clean) * page;
    m.compressed = double(vs.compressor_page_count) * page;
    m.wired = double(vs.wire_count) * page;
    m.swapouts = double(vs.swapouts);
    return m;
}

void log_line(FILE* f, const std::string& s)
{
    const auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    char ts[32];
    std::strftime(ts, sizeof(ts), "%H:%M:%S", std::localtime(&t));
    std::fprintf(f, "%s %s\n", ts, s.c_str());
    std::fflush(f);
    fsync(fileno(f));
}

} // namespace

int main(int argc, char** argv)
{
    std::string log_path;
    double min_avail = 8e9, max_compressed = 4e9;
    int i = 1;
    for (; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--")
            break;
        if (i + 1 >= argc)
            return 2;
        if (a == "--log")
            log_path = argv[++i];
        else if (a == "--min-avail-gb")
            min_avail = std::stod(argv[++i]) * 1e9;
        else if (a == "--max-compressed-gb")
            max_compressed = std::stod(argv[++i]) * 1e9;
        else
            return 2;
    }
    if (log_path.empty() || i + 1 >= argc) {
        std::fprintf(stderr, "usage: mem_watchdog --log FILE [--min-avail-gb G] [--max-compressed-gb G] -- cmd...\n");
        return 2;
    }
    FILE* log = std::fopen(log_path.c_str(), "a");
    if (!log)
        return 1;
    const Mem start = read_mem();
    log_line(log, "start: free " + std::to_string(start.free / 1e9) + " GB, avail " + std::to_string(start.avail / 1e9) +
                      " GB, compressed " + std::to_string(start.compressed / 1e9) + " GB, wired " +
                      std::to_string(start.wired / 1e9) + " GB");

    const pid_t child = fork();
    if (child == 0) {
        setpgid(0, 0);
        execvp(argv[i + 1], argv + i + 1);
        _exit(127);
    }
    setpgid(child, child);
    // The compressor limit is on growth: whatever was compressed before the run is not ours.
    const double compressed_base = start.compressed;
    int status = 0, tick = 0;
    for (;;) {
        const pid_t r = waitpid(child, &status, WNOHANG);
        if (r == child)
            break;
        const Mem m = read_mem();
        char buf[256];
        std::snprintf(buf, sizeof(buf), "free %.1f GB, avail %.1f GB, compressed %.1f GB, wired %.1f GB, swapouts %.0f",
                      m.free / 1e9, m.avail / 1e9, m.compressed / 1e9, m.wired / 1e9, m.swapouts);
        if (m.avail < min_avail || m.compressed - compressed_base > max_compressed) {
            kill(-child, SIGKILL);
            log_line(log, std::string("KILLED the run: ") + buf);
            waitpid(child, &status, 0);
            std::fclose(log);
            return 3;
        }
        if (++tick % 10 == 0)
            log_line(log, buf);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    log_line(log, "run finished, status " + std::to_string(WIFEXITED(status) ? WEXITSTATUS(status) : -1));
    std::fclose(log);
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}
