#!/usr/bin/env python3
"""Triage a KytyPS5 run log: what compiled, what was skipped, what killed it.
Usage: triage.py <log> [--json out.json] [--radv-dir <folder holding radv_dumps_*>]"""
import re, sys, json, collections, os, glob
path = sys.argv[1]
ts_re = re.compile(rb'time = ([0-9.]+)')
hash_re = re.compile(rb'hash=(0x[0-9a-f]{16})')
stage_re = re.compile(rb'ShaderRecompiler (\w\w) phase end: hash=(0x[0-9a-f]{16}) (.+?) (?:blocks=|elapsed)')
skip_re = re.compile(rb'!!! SHADER (?:\w+ )?SKIP \(([^)]*)\) shader=(0x[0-9a-f]{16})(?: stage=(\w+)(?: at (\S+?):)?)?(?:[^:]*: (.*))?')
uses_re = re.compile(rb'!!! SHADER SKIP USES shader=(0x[0-9a-f]{16}) count=(\d+)')
clock_re = re.compile(rb'^\[\d+\]\[(\d\d):(\d\d):(\d\d)\.(\d+)\]')
norm = lambda b: re.sub(rb'0x[0-9a-fA-F]+', b'0xN', re.sub(rb'\b\d+\b', b'N', b))
t = 0.0
trace_re = re.compile(rb'^\[(\d+)\]\[trace\] (\S+)\(\) #(\d+)')
calls = {}   # function -> [count, first guest time, last guest time]
compiled = {}            # hash -> (stage, first time seen)
last_phase = {}          # hash -> last finished phase
skips = collections.OrderedDict()
bang = collections.Counter(); unresolved_called = collections.Counter()
flips = timedout = lines = 0
flip_t = collections.Counter(); shader_t = collections.Counter()
driver_hang = []; fatal = []; device_lost = []; in_device_lost = False; tail = collections.deque(maxlen=40); first_fatal_line = None
with open(path, 'rb') as f:
    for line in f:
        lines += 1
        m = ts_re.search(line)
        if m and line.startswith(b'Controller'):
            continue
        if line.startswith(b'=== GPU DEVICE LOST TRIAGE ==='):
            in_device_lost = True
            device_lost.append(line.rstrip().decode(errors='replace'))
            continue
        if in_device_lost:
            device_lost.append(line.rstrip().decode(errors='replace')[:600])
            if line.startswith(b'=== END GPU DEVICE LOST TRIAGE ==='):
                in_device_lost = False
            continue
        if line.startswith((b'driver_dump:', b'radv: GPU hang', b'radv: can')):
            driver_hang.append(line.rstrip().decode(errors='replace')[:400])
        if line.startswith(b'Controller'): continue
        if line.startswith(b'@@T '):
            t = float(line[4:]); continue
        if line.startswith(b'[') and b'][trace] ' in line[:24]:
            m = trace_re.match(line)
            if m:
                c = calls.setdefault(m.group(2).decode(), [0, t, t]); c[0] = max(c[0], int(m.group(3))); c[2] = t
            continue
        if line.startswith(b'['):
            m = clock_re.match(line)
            if m and False: t = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + float(b'0.' + m.group(4))
        tail.append(line.rstrip()[:300])
        if b'ShaderRecompiler' in line:
            m = stage_re.search(line)
            if m:
                st, h, ph = m.group(1).decode(), m.group(2).decode(), m.group(3).decode()
                if h not in compiled:
                    compiled[h] = (st, t); shader_t[int(t // 10) * 10] += 1
                last_phase[h] = ph
            continue
        if line.startswith(b'!!!'):
            u = uses_re.search(line)
            m = None if u else skip_re.search(line)
            if u:
                if u.group(1).decode() in skips: skips[u.group(1).decode()]['uses'] = int(u.group(2))
            elif m:
                reason = (m.group(5) or m.group(1)).decode(errors='replace')
                reason = re.sub(r'^.*?pc[= ]0x[0-9a-f]+:? ', '', reason)
                reason = re.sub(r'raw=\[[^\]]*\] ?', '', re.sub(r'0x[0-9a-f]{8}: ', '', reason))
                skips.setdefault(m.group(2).decode(), {'kind': m.group(1).decode(), 'stage': (m.group(3) or b'').decode(),
                    'site': (m.group(4) or b'').decode().split('/src/')[-1], 'reason': reason, 't': round(t, 1), 'uses': 1})
            else:
                bang[norm(line.rstrip())[:160].decode(errors='replace')] += 1
            continue
        if b'Unresolved import stub called:' in line:
            unresolved_called[line.split(b'called:')[1].strip().decode(errors='replace')] += 1
        elif b'Equeue wait timedout: GPU Flip Event' in line: timedout += 1
        elif b'GPU Flip Event' in line and b'Equeue wait:' in line: flips += 1; flip_t[int(t // 10) * 10] += 1
        elif line.startswith(b'--- ') and (b'Error' in line or b'Guest fault' in line or b'Build' in line):
            if first_fatal_line is None: first_fatal_line = lines
            fatal.append(line.rstrip().decode(errors='replace'))
            for _ in range(70 if b'Guest fault' in line else 3):
                nx = f.readline()
                if not nx: break
                lines += 1
                if nx.startswith(b'--- Host backtrace'): break
                if not nx.startswith(b'Controller') and b'][trace]' not in nx[:24]: fatal.append(nx.rstrip().decode(errors='replace')[:600])
stages = collections.Counter(s for s, _ in compiled.values())
unfinished = {h: p for h, p in last_phase.items() if not p.startswith(('SPIR', 'Emit', 'Optimize'))}
phases = collections.Counter(last_phase.values())
out = dict(log=path, lines=lines, run_seconds=round(t, 1), shaders_seen=len(compiled), by_stage=dict(stages),
           last_phase_histogram=dict(phases.most_common(12)),
           new_shaders_per_10s=dict(sorted(shader_t.items())), flip_waits=flips, flip_wait_timeouts=timedout,
           flip_waits_per_10s=dict(sorted(flip_t.items())), skipped=skips, other_bang_lines=dict(bang.most_common(25)),
           unresolved_stubs_called=dict(unresolved_called), device_lost=device_lost, driver_hang=driver_hang, fatal=fatal)
if '--json' in sys.argv: json.dump(out, open(sys.argv[sys.argv.index('--json') + 1], 'w'), indent=1)
print(f"log {path}: {lines} lines, ~{out['run_seconds']} s of wall time")
print(f"shaders seen: {len(compiled)}  by stage: {dict(stages)}")
print("last finished phase per shader:", dict(phases.most_common(12)))
print("new shaders per 10 s:", out['new_shaders_per_10s'])
print(f"flip waits: {flips} (timeouts {timedout}); per 10 s: {out['flip_waits_per_10s']}")
print(f"skipped shaders: {len(skips)}")
classes = collections.defaultdict(list)
for h, v in skips.items(): classes[(v['site'], re.sub(r'(?<!opcode=)0x[0-9a-f]+|\b\d+\b', 'N', v['reason'])[:200])].append(h)
out['failure_classes'] = [dict(site=k[0], reason=k[1], shaders=len(v), uses=sum(skips[h]['uses'] for h in v), examples=v[:3])
                          for k, v in sorted(classes.items(), key=lambda kv: -len(kv[1]))]
print(f"failure classes: {len(classes)}  (one class = one fix; shaders / uses (>=, power-of-two counts) / site / reason)")
for c in out['failure_classes']: print(f"  {c['shaders']:4d} sh {c['uses']:8d} uses  {c['site']}  {c['reason'][:150]}  e.g. {c['examples'][0]}")
if '--json' in sys.argv: json.dump(out, open(sys.argv[sys.argv.index('--json') + 1], 'w'), indent=1)
print("other '!!!' lines:"); [print(f"  {n:6d} x {k}") for k, n in bang.most_common(25)]
out['calls'] = {k: dict(count=v[0], first=round(v[1], 1), last=round(v[2], 1)) for k, v in calls.items()}
if calls:
    libs = collections.Counter()
    for k, v in calls.items(): libs[k.split('::')[0]] += v[0]
    print(f"system calls traced: {len(calls)} functions (counts are lower bounds: 1-4, then powers of two)")
    print("  by library:", dict(libs.most_common(40)))
    late = sorted((v[1], k, v[0]) for k, v in calls.items())[-25:]
    print("  last 25 functions to appear (guest s, name, calls):")
    for ft, k, n in late: print(f"    {ft:7.1f}  {k}  x{n}")
    for lib in ('Pad', 'UserService', 'SystemService', 'SaveData', 'NpManager'):
        fs = {k.split('::')[-1]: v[0] for k, v in calls.items() if k.startswith(lib + '::')}
        print(f"  {lib}: {fs if fs else 'never called'}")
print("unresolved stubs called:", dict(unresolved_called))
if device_lost:
    print("GPU DEVICE LOST TRIAGE:")
    [print("  " + x) for x in device_lost]
# Driver (RADV_DEBUG=hang) dump folders: file list, pipeline.log size, the faulting-address line.
radv_dir = sys.argv[sys.argv.index('--radv-dir') + 1] if '--radv-dir' in sys.argv else None
dumps = sorted(glob.glob(os.path.join(radv_dir, 'radv_dumps_*'))) if radv_dir else []
if driver_hang or dumps:
    print("DRIVER HANG DUMP:")
    [print("  log: " + x) for x in driver_hang]
    out['driver_dumps'] = []
    for d in dumps:
        names = sorted(os.listdir(d)); pl = os.path.join(d, 'pipeline.log')
        fault = []
        for n in ('vm_fault.log', 'dmesg.log'):
            p = os.path.join(d, n)
            if os.path.isfile(p):
                for l in open(p, errors='replace'):
                    if re.search(r'fault|address', l, re.I) and l.strip(): fault.append(f"{n}: {l.strip()[:200]}")
        out['driver_dumps'].append(dict(path=d, files=names, pipeline_log_bytes=os.path.getsize(pl) if os.path.isfile(pl) else None, fault_lines=fault[:6]))
        print(f"  {d}: {len(names)} files: {' '.join(names)}")
        print(f"  pipeline.log: {os.path.getsize(pl) if os.path.isfile(pl) else 'missing'} bytes")
        [print("  " + x) for x in fault[:6]]
    if not dumps: print("  no radv_dumps_* folder found")
    if '--json' in sys.argv: json.dump(out, open(sys.argv[sys.argv.index('--json') + 1], 'w'), indent=1)
print("FATAL:" if fatal else "no fatal block"); [print("  " + x) for x in fatal]
