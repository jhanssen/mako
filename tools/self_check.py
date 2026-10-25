"""Checks a mako build on this Mac and prints a Markdown report to paste into a GitHub issue.

    python3 tools/self_check.py --model <model-00001-of-0000N.gguf> [--build build] [--quick] [--out report.md]

Runs the test suite against the model (--quick: only the tests that need no GPU work), then a fixed benchmark:
prefill of 512 and 4096 new tokens on a 4K-token context, and decode of 128 tokens with and without the
next-token block's drafts. Each step loads the model, and waits first until memory is free again.
"""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path


def run(cmd, env=None, timeout=1800):
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def sysctl(name):
    code, out = run(["sysctl", "-n", name])
    return out.strip() if code == 0 else "?"


def free_gb():
    code, out = run(["vm_stat"])
    if code != 0:
        return 0.0
    page = int(re.search(r"page size of (\d+)", out).group(1))
    pages = 0
    for key in ("Pages free", "Pages speculative"):
        m = re.search(key + r":\s+(\d+)", out)
        pages += int(m.group(1)) if m else 0
    return pages * page / 1e9


def wait_for_memory(log):
    # A process that just exited can take a while to return its memory (a model loaded before that gets a small
    # expert cache): wait until free memory stops growing.
    history = [free_gb()]
    for _ in range(60):
        time.sleep(1)
        history.append(free_gb())
        if len(history) >= 4 and history[-1] - history[-4] < 1.0:
            return
    log(f"note: free memory still growing after a minute ({history[-1]:.0f} GB); results may be slower")


def machine():
    gpu_cores = "?"
    code, out = run(["system_profiler", "SPDisplaysDataType", "-json"])
    if code == 0:
        try:
            gpu = json.loads(out)["SPDisplaysDataType"][0]
            gpu_cores = gpu.get("sppci_cores", "?")
        except (KeyError, IndexError, ValueError):
            pass
    return {
        "chip": sysctl("machdep.cpu.brand_string"),
        "model": sysctl("hw.model"),
        "cores": f'{sysctl("hw.perflevel0.physicalcpu")}P + {sysctl("hw.perflevel1.physicalcpu")}E',
        "gpu_cores": gpu_cores,
        "ram_gb": int(sysctl("hw.memsize")) / 1e9 if sysctl("hw.memsize").isdigit() else 0,
        "macos": f"{platform.mac_ver()[0]} ({sysctl('kern.osversion')})",
        "wired_limit_mb": sysctl("iogpu.wired_limit_mb"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=os.environ.get("MAKO_MODEL", ""), help="first GGUF shard (or MAKO_MODEL)")
    ap.add_argument("--build", default="build")
    ap.add_argument("--quick", action="store_true", help="skip the GPU tests and the benchmark")
    ap.add_argument("--out", default="", help="also write the report to this file")
    a = ap.parse_args()
    log = lambda s: print(s, file=sys.stderr, flush=True)
    root = Path(__file__).resolve().parent.parent
    build = (root / a.build) if not Path(a.build).is_absolute() else Path(a.build)
    if not a.model or not Path(a.model).exists():
        sys.exit("give --model <first GGUF shard> (or set MAKO_MODEL)")
    if not (build / "tools" / "prefill_sweep").exists():
        sys.exit(f"no build in {build}; build first (cmake -S . -B build && cmake --build build)")

    m = machine()
    code, commit = run(["git", "-C", str(root), "describe", "--always", "--dirty"])
    commit = commit.strip() if code == 0 else "?"
    env = dict(os.environ, MAKO_MODEL=str(Path(a.model).resolve()))

    log("running the tests...")
    wait_for_memory(log)
    ctest = ["ctest", "--test-dir", str(build), "--output-on-failure"]
    if a.quick:
        ctest += ["-L", "tokenizer|sampler|server"]
    started = time.time()
    _, out = run(ctest, env=env, timeout=3600)
    tests = re.findall(r"Test\s+#\d+: (\S+) \.+\s*(\*{3}\w+|Passed|\w+)\s+([\d.]+) sec", out)
    failed_output = out[out.find("The following tests FAILED"):] if "The following tests FAILED" in out else ""
    tests_s = time.time() - started

    bench = {}
    if not a.quick:
        sweep = str(build / "tools" / "prefill_sweep")
        log("benchmark: prefill...")
        wait_for_memory(log)
        _, out_p = run([sweep, a.model, "--base", "4000", "--sizes", "512,4096", "--repeats", "1"])
        for n, ms, tps in re.findall(r"^\| (\d+) \| 1 \| (\d+) \| (\d+) \|", out_p, re.M):
            bench[f"prefill {n} new tokens on a 4K context"] = f"{tps} tok/s ({ms} ms)"
        for spec in (True, False):
            log(f"benchmark: decode {'with' if spec else 'without'} drafts...")
            wait_for_memory(log)
            cmd = [sweep, a.model, "--base", "4000", "--decode", "128"] + ([] if spec else ["--no-spec"])
            _, out_d = run(cmd)
            mt = re.search(r"\(([\d.]+) ms/token\).*kept \((\d+)%\)", out_d)
            if mt:
                label = "decode 128 tokens, drafts" if spec else "decode 128 tokens, no drafts"
                rate = 1000 / float(mt.group(1))
                bench[label] = f"{rate:.1f} tok/s ({mt.group(1)} ms/token)" + (f", {mt.group(2)}% of drafts kept" if spec else "")

    lines = ["## mako self-check", "",
             f"- mako `{commit}`, model `{Path(a.model).name}`",
             f"- {m['chip']} ({m['model']}), {m['cores']} CPU cores, {m['gpu_cores']} GPU cores, {m['ram_gb']:.0f} GB RAM, "
             f"macOS {m['macos']}, GPU wired limit {m['wired_limit_mb'] if m['wired_limit_mb'] not in ('0', '?') else 'default'}",
             "", "| test | result | seconds |", "|---|---|---|"]
    for name, result, secs in tests:
        lines.append(f"| {name} | {result.strip('*')} | {secs} |")
    if not tests:
        lines.append("| (no test results) | | |")
    passed = sum(1 for _, r, _ in tests if r == "Passed")
    lines += ["", f"{passed} of {len(tests)} tests passed in {tests_s:.0f} s."]
    if bench:
        lines += ["", "| benchmark | result |", "|---|---|"] + [f"| {k} | {v} |" for k, v in bench.items()]
    if failed_output:
        lines += ["", "<details><summary>failures</summary>", "", "```", failed_output.strip()[-4000:], "```", "",
                  "</details>"]
    report = "\n".join(lines) + "\n"
    print(report)
    if a.out:
        Path(a.out).write_text(report)


if __name__ == "__main__":
    main()
