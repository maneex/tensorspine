#!/usr/bin/env python3
"""Run a command and report its peak resident memory, sampled from /proc every half
second: VmRSS, and RssAnon (what the process allocated) apart from RssFile (memory-mapped
weights the kernel may drop at any time). The evidence for `--max-ram`.

    python3 fixtures/peak_rss.py -- python3 ref.py run … --max-ram 8
"""
import subprocess
import sys
import time


def sample(pid):
    out = {}
    try:
        with open(f"/proc/{pid}/status", encoding='utf-8') as f:
            for line in f:
                k = line.split(':')[0]
                if k in ('VmRSS', 'RssAnon', 'RssFile'):
                    out[k] = int(line.split()[1]) // 1024
    except OSError:
        pass
    return out


def main(argv):
    cmd = argv[argv.index('--') + 1:] if '--' in argv else argv
    proc = subprocess.Popen(cmd)
    peak = {'VmRSS': 0, 'RssAnon': 0, 'RssFile': 0}
    t0 = time.time()
    while proc.poll() is None:
        s = sample(proc.pid)
        for k in peak:
            peak[k] = max(peak[k], s.get(k, 0))
        time.sleep(0.5)
    print(f"peak resident: VmRSS {peak['VmRSS'] / 1024:.2f} GiB — RssAnon {peak['RssAnon'] / 1024:.2f} GiB allocated, "
          f"RssFile {peak['RssFile'] / 1024:.2f} GiB memory-mapped — over {time.time() - t0:.0f}s; exit {proc.returncode}")
    return proc.returncode


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
