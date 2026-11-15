// arena_budget: the file-cache floor predicted for after the budget is wired, not read from the kernel's last scan.
#include "core/arena.hpp"

#include <cmath>
#include <format>
#include <print>
#include <string>

using mako::core::MemInfo;

namespace {

constexpr double GB = 1e9;
int failures = 0;

void check(bool ok, const std::string& what)
{
    std::println("  {}  {}", ok ? "ok  " : "FAIL", what);
    failures += !ok;
}

// A 128 GB Mac after the dense weights are loaded.
MemInfo mac128()
{
    MemInfo m;
    m.phys = 137.44 * GB;
    m.working_set = 115 * GB;
    m.footprint = 9.5 * GB;
    m.pageable = 107 * GB;
    m.available = 120 * GB;
    return m;
}

} // namespace

int main()
{
    {
        // The sysctl as an idle machine's last scan left it (43.3 GB) and as a loaded one's did (27 GB): the same
        // budget, a quarter of RAM's.
        MemInfo idle = mac128(), loaded = mac128();
        idle.filecache_min = 43.3 * GB;
        loaded.filecache_min = 27 * GB;
        const double quarter = 0.75 * idle.phys - idle.footprint;
        const double a = mako::core::arena_budget(idle), b = mako::core::arena_budget(loaded);
        check(std::abs(a - quarter) < 1e6, std::format("stale idle floor: {:.2f} GB, a quarter of RAM gives {:.2f}",
                                                       a / GB, quarter / GB));
        check(a == b, std::format("idle and loaded readings agree: {:.2f} / {:.2f} GB", a / GB, b / GB));
    }
    {
        // A sysctl above 10/27 of RAM means a smaller divisor: its ratio binds, and the budget leaves exactly the
        // floor that ratio predicts for after it is wired, plus 12 GB.
        MemInfo m = mac128();
        m.pageable = 127 * GB;
        m.filecache_min = 0.8 * m.phys;
        const double r = 0.8, budget = mako::core::arena_budget(m);
        const double floor = (m.pageable - budget) * r + 12 * GB;
        check(budget < 0.75 * m.phys - m.footprint, std::format("divisor guard binds: {:.2f} GB", budget / GB));
        check(std::abs(m.phys - m.footprint - budget - floor) < 1e6,
              std::format("footprint + budget + floor = RAM: {:.2f} + {:.2f} + {:.2f} = {:.2f} GB", m.footprint / GB,
                          budget / GB, floor / GB, m.phys / GB));
    }
    {
        // On this machine: the pageable pages are measured, and the kernel's reading fits divisor 27 (a value above
        // 10/27 of RAM is impossible with it).
        const MemInfo m = mako::core::query_memory(0);
        check(m.pageable > 0 && m.pageable <= m.phys,
              std::format("pageable {:.1f} GB of {:.1f} GB RAM", m.pageable / GB, m.phys / GB));
        check(m.filecache_min <= m.phys * 10 / 27,
              std::format("kernel floor {:.1f} GB within 10/27 of RAM ({:.1f} GB)", m.filecache_min / GB,
                          m.phys * 10 / 27 / GB));
    }
    if (failures)
        std::println("{} failed", failures);
    return failures ? 1 : 0;
}
