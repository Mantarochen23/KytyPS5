#!/usr/bin/env python3
"""Triage a KytyPS5 run log: every finding of a boot, in one streaming pass.
Usage: triage.py <log> [--json out.json] [--radv-dir <folder holding radv_dumps_*>]"""
import re, sys, json, collections, os, glob, statistics, heapq
path = sys.argv[1]
ts_re = re.compile(rb'time = ([0-9.]+)')
hash_re = re.compile(rb'hash\s*=\s*(0x[0-9a-f]{16})')
stage_re = re.compile(rb'ShaderRecompiler (\w\w) phase end: hash\s*=\s*(0x[0-9a-f]{16}) (.+?) (?:blocks=|elapsed)')
elapsed_re = re.compile(rb'elapsed_ms=(\d+)')
outcome_re = re.compile(rb'ShaderRecompiler (\w\w) translation end: hash\s*=\s*(\S+) outcome=(.*)$')
# New line kinds. All are optional: their absence simply leaves the matching
# section empty. Hashes in tests are pseudonyms; no real ids live in this file.
build_re = re.compile(rb'^\[build\] \w+ hash=(0x[0-9a-f]{16})\b.*?translate=(\d+)ms '
                      rb'.*?emit=(\d+)ms .*?module=(\d+)ms .*?pipeline=(\d+)ms .*?total=(\d+)ms')
build_busy_re = re.compile(rb'workers (\d+)/\d+ busy')
park_count_re = re.compile(rb'^\[park\] (\d+) guest threads parked')
park_thread_re = re.compile(rb'^\[park\] .*?\(id (\d+)\) parked for good: pc=(0x[0-9a-fA-F]+)'
                            rb'(?: = (\S+))?')
shader_id_re = re.compile(rb'(?:shader|hash)\s*=\s*(0x[0-9a-f]{16})')
miss_field_re = re.compile(rb': ([^:]+?)\s+(\d+) vs (\d+)')
miss_no_re = re.compile(rb'miss #(\d+)')
skip_re = re.compile(rb'!!! SHADER (?:\w+ )?SKIP \(([^)]*)\) shader\s*=\s*(0x[0-9a-f]{16})(?: stage=(\w+)(?: at (\S+?):)?)?(?:[^:]*: (.*))?')
uses_re = re.compile(rb'!!! SHADER SKIP USES shader\s*=\s*(0x[0-9a-f]{16}) count=(\d+)')
clock_re = re.compile(rb'^\[\d+\]\[(\d\d):(\d\d):(\d\d)\.(\d+)\]')
norm = lambda b: re.sub(rb'0x[0-9a-fA-F]+', b'0xN', re.sub(rb'\b\d+\b', b'N', b))
ansi_re = re.compile(rb'^\x1b\[[0-9;]*m')
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

# Findings. All of these are bounded: the log is read once, line by line, and no
# line buffer grows with the size of the log.
# `[fault]` lines (three per 5 s interval): rates of the write-watch path. Bounded: at most 2000
# intervals are kept; block totals are a Counter capped at 4000 keys.
fault_re = re.compile(rb'^\[fault\] ([\d.]+)s write=([\d.]+)/s read=([\d.]+)/s armed=([\d.]+) pg/s '
                      rb'released=([\d.]+) pg/s protect-calls=([\d.]+)/s protect=([\d.]+) ms/s '
                      rb'handler=([\d.]+) ms/s invalidate=([\d.]+) ms/s fast-skip=([\d.]+)/s '
                      rb'storm\(on=(\d+)\) marked=(\d+) kept=([\d.]+) pg/s expired=(\d+)')
fault_wait_re = re.compile(rb'(\w+)=([\d.]+)\(([\d.]+)\)')
fault_block_re = re.compile(rb'(0x[0-9a-f]+)=(\d+)')
fault_rows = []; fault_waits = collections.Counter(); fault_blocks = collections.Counter()
frame_count = 0; frame_t = collections.Counter(); last_frame_t = None; last_frame_no = None
gaps = []                 # frame gaps >= 3 s: start, end, length, frame number
stalls = collections.OrderedDict()   # (seconds, frame) -> phases + times
queue_suspends = {}       # queue -> longest single-shader suspension
mem_waits = {}            # (queue, kind, address, compare, wanted, seen) -> worst wait
mem_wait_lines = 0
outcomes = {}             # (stage, outcome) -> [count, first t, last t]
translations = {}         # hash -> [count, up to 16 times]
phase_stats = {}          # hash -> [total ms, worst ms, worst phase, stage, first t, last t]
render_built = [0, 0.0, 0, None]   # total ms, worst ms, count, time of worst
async_count = 0; peak_queued = (0, 0.0); peak_waiting = (0, 0.0)
jobs_samples = []; worst_pump = (0.0, 0.0); pump_peak = (0.0, 0.0); stages_tail = None
cache_lines = []; last_store = None; store_lookups = None
skip_lines = 0; skip_t_first = None; skip_t_last = None
ledger_lines = []; ledger_t = None

# diag2 findings. Every collection is bounded so a large log cannot grow memory
# without limit; only the first/last examples and the largest builds are kept.
loading_gap = None              # first frame gap >= 20 s (end of loading)
post_loading_frames = []        # (t, frame no, gap before it), capped
window_finishes = {}            # 60 s window -> [CS emitted times]
window_busy = {}                # 60 s window -> peak workers busy
build_totals = collections.Counter()
build_top = []                  # heap: (total ms, sequence, record); 10 largest
build_seq = 0
park_threads = []               # parked thread records, capped
park_summary = []               # (t, count) from '[park] N guest threads parked'
holes = {}                      # normalised failure text -> grouped record
store_miss_fields = {}          # normalised differing field -> grouped record
module_bases = []               # (base address, name-or-?) for pc attribution
last_shader_id = None
parse_errors = 0

frame_re = re.compile(rb'flip_arg\s*=\s*(\d+)')
stall_re = re.compile(rb'^\[stall\] no frame for ([\d.]+) s \(frame (\d+)\): render thread in \'([^\']*)\'\s*(.*?)\s*for ([\d.]+) s; shader hash\s*=\s*(\S+) stage=([^;\s]+)')
queue_re = re.compile(rb'^\[wait\] queue (\S+) \(([^)]+)\): (.*)$')
suspend_re = re.compile(rb'suspended for shader (\S+) for ([\d.]+) s')
memwait_re = re.compile(rb'(\w+) on memory (\S+) mask (\S+) compare (.*?), value at last poll (\S+);.*?polled (\d+) times in ([\d.]+) s')
built_re = re.compile(rb'^Async STALL: the render thread built (\d+) ms in ([\d.]+) s \(worst (\d+) ms\)')
a_queued_re = re.compile(rb'(\d+) queued'); a_waiting_re = re.compile(rb', (\d+) waiting \(')
a_jobs_re = re.compile(rb'([\d.]+) jobs/s'); a_pump_re = re.compile(rb'pump ([\d.]+) ms \(worst ([\d.]+) ms')
store_kv_re = re.compile(rb'([\w-]+)=(\d+)')   # keys such as miss-key, distinct-miss
stage_kv_re = re.compile(r'(\w+)=(\d+)')
lead_re = re.compile(r'lead time ms \(use - registration\) (\S+): n=(\d+)(?: min=(\d+))?')


def module_for_pc(pc):
    """Best module+offset guess from base_vaddr lines, or '?+0x..'."""
    value = int(pc, 16)
    base, name = max(module_bases, default=(None, None))
    if base is None or value < base:
        return '?+%s' % pc
    return '%s+0x%x' % (name or '?', value - base)


def add_hole(line, t):
    """One dispatch skip or failed stage, grouped by normalised failure text."""
    global last_shader_id
    sid = shader_id_re.search(line)
    if sid:
        last_shader_id = sid.group(1).decode()
    if line.startswith(b'!!!'):
        paren = re.search(rb'\(([^)]*)\)', line)
        text = (paren.group(1) if paren else line[line.find(b'SKIP') + 4:]).strip()
        if b';' in line:
            text = (text + b';' + line.split(b';', 1)[1]).strip()
    else:
        text = line.strip()
    key = norm(text)[:200].decode(errors='replace')
    e = holes.get(key)
    if e is None and len(holes) < 500:
        e = holes[key] = dict(reason=key, count=0, shaders=set(),
                              t_first=round(t, 1), t_last=round(t, 1), examples=[])
    if e is not None:
        e['count'] += 1; e['t_last'] = round(t, 1)
        if last_shader_id and len(e['shaders']) < 2000:
            e['shaders'].add(last_shader_id)
        if len(e['examples']) < 3:
            e['examples'].append(line.rstrip().decode(errors='replace')[:200])


def add_build(line, t):
    """Accumulate [build] timings; keep only the ten largest totals."""
    global build_seq
    m = build_re.match(line)
    if not m:
        return
    h = m.group(1).decode()
    fields = dict(translate=int(m.group(2)), emit=int(m.group(3)),
                  module=int(m.group(4)), pipeline=int(m.group(5)))
    total = int(m.group(6))
    for key, value in fields.items():
        build_totals[key] += value
    rec = dict(hash=h, total=total, t=round(t, 1), **fields)
    entry = (total, build_seq, rec); build_seq += 1
    if len(build_top) < 10:
        heapq.heappush(build_top, entry)
    elif total > build_top[0][0]:
        heapq.heapreplace(build_top, entry)


def add_miss_field(line, t):
    """Group a '[store] miss #..' line by the field that differed."""
    m = miss_field_re.search(line)
    field = norm(m.group(1)).strip().decode(errors='replace') if m else '?'
    sid = shader_id_re.search(line)
    e = store_miss_fields.get(field)
    if e is None and len(store_miss_fields) < 500:
        e = store_miss_fields[field] = dict(field=field, count=0, shaders=set(),
                                            t_first=round(t, 1), t_last=round(t, 1))
    if e is not None:
        e['count'] += 1; e['t_last'] = round(t, 1)
        sid_text = sid.group(1).decode() if sid else None
        if sid_text and len(e['shaders']) < 2000:
            e['shaders'].add(sid_text)


with open(path, 'rb') as f:
    for line in f:
        try:
            lines += 1
            if line[:1] == b'\x1b':
                line = ansi_re.sub(b'', line, count=1)
            c = line[:1]
            if c == b'b' and line.startswith(b'base_vaddr'):
                bm = re.search(rb'=\s*(0x[0-9a-fA-F]+)', line)
                if bm and len(module_bases) < 200:
                    module_bases.append((int(bm.group(1), 16), '?'))
            if c == b'C' and line.startswith(b'Controller'): continue
            if c == b'=' and line.startswith(b'=== GPU DEVICE LOST TRIAGE ==='):
                in_device_lost = True
                device_lost.append(line.rstrip().decode(errors='replace'))
                continue
            if in_device_lost:
                device_lost.append(line.rstrip().decode(errors='replace')[:600])
                if line.startswith(b'=== END GPU DEVICE LOST TRIAGE ==='):
                    in_device_lost = False
                continue
            if c == b'd' and line.startswith(b'driver_dump:') or c == b'r' and line.startswith((b'radv: GPU hang', b'radv: can')):
                driver_hang.append(line.rstrip().decode(errors='replace')[:400])
            if c == b'@' and line.startswith(b'@@T '):
                t = float(line[4:]); continue
            if c == b'[' and line.startswith(b'[fault] '):
                m = fault_re.match(line)
                if m:
                    if len(fault_rows) < 2000:
                        fault_rows.append([round(t, 1)] + [float(x) for x in m.groups()])
                elif line.startswith(b'[fault] lock-wait'):
                    for name, ms, n in fault_wait_re.findall(line):
                        fault_waits[name.decode()] += float(ms)
                elif line.startswith(b'[fault] top 64K blocks:'):
                    for addr, n in fault_block_re.findall(line):
                        if addr in fault_blocks or len(fault_blocks) < 4000:
                            fault_blocks[addr] += int(n)
                continue
            if c == b'[' and b'][trace] ' in line[:24]:
                m = trace_re.match(line)
                if m:
                    c = calls.setdefault(m.group(2).decode(), [0, t, t]); c[0] = max(c[0], int(m.group(3))); c[2] = t
                continue
            if b'flip_arg' in line:
                m = frame_re.search(line)
                if m:
                    frame_count += 1; frame_t[int(t // 10) * 10] += 1
                    gap = 0.0 if last_frame_t is None else t - last_frame_t
                    if last_frame_t is not None and gap >= 3.0 and len(gaps) < 1000:
                        gaps.append(dict(start=round(last_frame_t, 1), end=round(t, 1),
                                         seconds=round(gap, 1), frame=int(m.group(1))))
                    if last_frame_t is not None and gap >= 20.0 and loading_gap is None:
                        loading_gap = dict(start=round(last_frame_t, 1), end=round(t, 1),
                                           seconds=round(gap, 1), frame=int(m.group(1)))
                    last_frame_t = t; last_frame_no = int(m.group(1))
                    if loading_gap is not None and len(post_loading_frames) < 200:
                        post_loading_frames.append((round(t, 1), last_frame_no, round(gap, 1)))
            if c == b'[':
                if line.startswith(b'[stall]'):
                    m = stall_re.match(line)
                    if m:
                        gap = (float(m.group(1)), int(m.group(2)))
                        g = stalls.setdefault(gap, dict(seconds=m.group(1).decode(), frame=m.group(2).decode(),
                                                        phases=[], t_first=round(t, 1), t_last=round(t, 1)))
                        g['t_last'] = round(t, 1)
                        if len(g['phases']) < 200:
                            g['phases'].append(dict(phase=m.group(3).decode(), state=m.group(4).decode(),
                                                    seconds=float(m.group(5)), hash=m.group(6).decode(),
                                                    stage=m.group(7).decode(), t=round(t, 1)))
                    elif len(stalls) < 1000:
                        stalls.setdefault(('?', lines), dict(seconds='?', frame='?', phases=[],
                                                              raw=line.rstrip().decode(errors='replace')[:300],
                                                              t_first=round(t, 1), t_last=round(t, 1)))
                    continue
                if line.startswith(b'[wait] queue'):
                    m = queue_re.match(line)
                    if m:
                        q = m.group(1).decode(); body = m.group(3)
                        u = suspend_re.search(body)
                        if u:
                            sec = float(u.group(2))
                            old = queue_suspends.get(q)
                            if old is None or sec > old['seconds']:
                                queue_suspends[q] = dict(seconds=sec, hash=u.group(1).decode(), t=round(t, 1))
                        else:
                            w = memwait_re.search(body)
                            if w and float(w.group(7)) >= 5.0:
                                mem_wait_lines += 1
                                compare = w.group(4).decode(errors='replace')
                                wanted = compare.split()[-1] if compare.split() else '?'
                                key = (q, w.group(1).decode(), w.group(2).decode(), compare,
                                       wanted, w.group(5).decode(errors='replace'))
                                e = mem_waits.get(key)
                                if e is None and len(mem_waits) < 500:
                                    e = mem_waits[key] = dict(queue=q, kind=w.group(1).decode(), address=w.group(2).decode(),
                                                              compare=compare, wanted=wanted, seen=w.group(5).decode(errors='replace'),
                                                              count=0, seconds=0.0, polls=0, t=round(t, 1))
                                if e is not None:
                                    e['count'] += 1
                                    if float(w.group(7)) > e['seconds']:
                                        e['seconds'] = float(w.group(7)); e['polls'] = int(w.group(6)); e['t'] = round(t, 1)
                                    e['polls'] = max(e['polls'], int(w.group(6)))
                    continue
                if line.startswith(b'[store]'):
                    last_store = (line.rstrip().decode(errors='replace'), round(t, 1))
                    if b'lookups=' in line:
                        store_lookups = dict((k.decode(), int(v)) for k, v in store_kv_re.findall(line))
                        store_lookups['t'] = round(t, 1)
                    if miss_no_re.search(line) and b' vs ' in line:
                        add_miss_field(line, t)
                    continue
                if line.startswith(b'[shader-reg]'):
                    ledger_t = round(t, 1)
                    if len(ledger_lines) < 100:
                        ledger_lines.append(line.rstrip().decode(errors='replace'))
                    continue
                if line.startswith(b'[park]'):
                    cm = park_count_re.match(line)
                    if cm:
                        if len(park_summary) < 200:
                            park_summary.append((round(t, 1), int(cm.group(1))))
                    else:
                        pm = park_thread_re.match(line)
                        if pm and len(park_threads) < 500:
                            pc = pm.group(2).decode()
                            modoff = (pm.group(3).decode(errors='replace') if pm.group(3)
                                      else module_for_pc(pc))
                            park_threads.append(dict(t=round(t, 1), id=int(pm.group(1)),
                                                     pc=pc, module=modoff))
                    continue
                if line.startswith(b'[build]'):
                    add_build(line, t)
                    continue
            if c == b'S' and line.startswith(b'Shader cache: loaded'):
                cache_lines.append((line.rstrip().decode(errors='replace'), round(t, 1)))
                continue
            if c == b'A' and line.startswith(b'Async STALL:'):
                m = built_re.match(line)
                if m:
                    ms, worst = int(m.group(1)), int(m.group(3))
                    render_built[0] += ms; render_built[1] = max(render_built[1], worst)
                    render_built[2] += 1
                    if worst == render_built[1]: render_built[3] = round(t, 1)
                continue
            if c == b'A' and line.startswith(b'Async:'):
                async_count += 1
                q = a_queued_re.search(line); w = a_waiting_re.search(line)
                j = a_jobs_re.search(line); p = a_pump_re.search(line)
                qn = int(q.group(1)) if q else 0
                if q and qn > peak_queued[0]: peak_queued = (qn, round(t, 1))
                if w and int(w.group(1)) > peak_waiting[0]: peak_waiting = (int(w.group(1)), round(t, 1))
                if j and qn > 0:
                    jobs_samples.append((float(j.group(1)), round(t, 1)))
                if p:
                    pump, worst = float(p.group(1)), float(p.group(2))
                    if worst > worst_pump[0]: worst_pump = (worst, round(t, 1))
                    if pump > pump_peak[0]: pump_peak = (pump, round(t, 1))
                if b'; stages: ' in line: stages_tail = line.split(b'; stages: ', 1)[1].decode(errors='replace')
                wnd = int(t // 60)
                if len(window_busy) < 2000:
                    bb = build_busy_re.search(line)
                    if bb:
                        busy = int(bb.group(1))
                        window_busy[wnd] = max(busy, window_busy.get(wnd, -1))
                    window_finishes.setdefault(wnd, [])
                continue
            if c == b'S' and b'ShaderRecompiler' in line:
                m = stage_re.search(line)
                if m:
                    st, h, ph = m.group(1).decode(), m.group(2).decode(), m.group(3).decode()
                    if h not in compiled:
                        compiled[h] = (st, t); shader_t[int(t // 10) * 10] += 1
                    last_phase[h] = ph
                    em = elapsed_re.search(line)
                    if em:
                        ms = int(em.group(1))
                        s = phase_stats.setdefault(h, [0, 0, '', st, round(t, 1), round(t, 1)])
                        s[0] += ms; s[5] = round(t, 1)
                        if ms > s[1]: s[1] = ms; s[2] = ph
                    continue
                if b'translation end' in line:
                    m = outcome_re.search(line)
                    if m:
                        st, h, outcome = m.group(1).decode(), m.group(2).decode(errors='replace'), m.group(3).decode(errors='replace').strip()
                        o = outcomes.setdefault((st, outcome), [0, round(t, 1), round(t, 1)])
                        o[0] += 1; o[2] = round(t, 1)
                        # Only a real build counts as a translation. A store hit
                        # is a reuse, not a translation of this shader.
                        if outcome.startswith('emitted'):
                            tr = translations.setdefault(h, [0, []])
                            tr[0] += 1
                            if len(tr[1]) < 16: tr[1].append(round(t, 1))
                            if st == 'CS' and len(window_finishes) < 2000:
                                wnd = int(t // 60)
                                times = window_finishes.setdefault(wnd, [])
                                if len(times) < 100000:
                                    times.append(t)
                continue
            if c == b'!' and line.startswith(b'!!!'):
                if b'SKIP' in line and b'USES' not in line:
                    add_hole(line, t)
                if line.startswith(b'!!! SHADER SKIP'):
                    skip_lines += 1
                    if skip_t_first is None: skip_t_first = round(t, 1)
                    skip_t_last = round(t, 1)
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
            if c == b's' and b'shader' in line and b'failed' in line:
                add_hole(line, t)
                continue
            if b'Unresolved import stub called:' in line:
                unresolved_called[line.split(b'called:')[1].strip().decode(errors='replace')] += 1
            elif b'Equeue wait timedout: GPU Flip Event' in line: timedout += 1
            elif b'GPU Flip Event' in line and b'Equeue wait:' in line: flips += 1; flip_t[int(t // 10) * 10] += 1
            elif c == b'-' and line.startswith(b'--- ') and (b'Error' in line or b'Guest fault' in line or b'Build' in line):
                if first_fatal_line is None: first_fatal_line = lines
                fatal.append(line.rstrip().decode(errors='replace'))
                for _ in range(70 if b'Guest fault' in line else 3):
                    nx = f.readline()
                    if not nx: break
                    lines += 1
                    if nx.startswith(b'--- Host backtrace'): break
                    if not nx.startswith(b'Controller') and b'][trace]' not in nx[:24]: fatal.append(nx.rstrip().decode(errors='replace')[:600])

        except Exception as exc:
            parse_errors += 1
            continue
def kv(line):
    return dict((k, int(v)) for k, v in stage_kv_re.findall(line))

stages = collections.Counter(s for s, _ in compiled.values())
unfinished = {h: p for h, p in last_phase.items() if not p.startswith(('SPIR', 'Emit', 'Optimize'))}
phases = collections.Counter(last_phase.values())

# ---- findings ---------------------------------------------------------------
# ---- findings ---------------------------------------------------------------
findings = {}
timeline = dict(last_frame_t=last_frame_t, frames=frame_count,
                frames_per_10s=dict(sorted(frame_t.items())), gaps=gaps,
                loading_gap=loading_gap,
                frames_after_loading=[dict(t=a, frame=b, gap=c)
                                      for a, b, c in post_loading_frames],
                parse_errors=parse_errors, t=round(t, 1))
findings['timeline'] = timeline
findings['fault_cost'] = dict(intervals=len(fault_rows), blocks={a.decode(): c for a, c in fault_blocks.most_common(8)},
                              lock_wait_ms_total=dict(fault_waits))

stall_list = []
for (secs, frame), g in sorted(stalls.items(), key=lambda kv: (-(float(kv[0][0]) if str(kv[0][0]).replace('.', '', 1).isdigit() else -1.0), str(kv[0][1]))):
    item = dict(seconds=g['seconds'], frame=g['frame'], t_first=g['t_first'], t_last=g['t_last'],
                phases=g['phases'])
    if 'raw' in g: item['raw'] = g['raw']
    stall_list.append(item)
findings['stalls'] = stall_list

findings['queues'] = dict(suspends=[dict(queue=q, **v) for q, v in sorted(queue_suspends.items(), key=lambda kv: -kv[1]['seconds'])],
                          memory_waits=[v for v in sorted(mem_waits.values(), key=lambda v: -v['seconds'])],
                          memory_wait_lines=mem_wait_lines, t=round(t, 1))

slowest = sorted(((v[0], h, v) for h, v in phase_stats.items()), reverse=True)[:20]
findings['shaders'] = dict(outcomes=[dict(stage=k[0], outcome=k[1], count=v[0], t_first=v[1], t_last=v[2])
                                     for k, v in sorted(outcomes.items(), key=lambda kv: -kv[1][0])],
                           slowest=[dict(hash=h, stage=v[3], total_ms=v[0], worst_ms=v[1], worst_phase=v[2],
                                         t_first=v[4], t_last=v[5]) for _, h, v in slowest],
                           repeated=[dict(hash=h, count=v[0], times=v[1]) for h, v in
                                     sorted(translations.items(), key=lambda kv: -kv[1][0]) if v[0] > 1][:20],
                           render_built=dict(total_ms=render_built[0], worst_ms=render_built[1],
                                             count=render_built[2], t_worst=render_built[3]), t=round(t, 1))

jobs_sorted = sorted(j for j, _ in jobs_samples)
findings['async'] = dict(peak_queued=dict(count=peak_queued[0], t=peak_queued[1]),
                         peak_waiting=dict(count=peak_waiting[0], t=peak_waiting[1]),
                         jobs_per_second=dict(min=jobs_sorted[0] if jobs_sorted else None,
                                              median=statistics.median(jobs_sorted) if jobs_sorted else None,
                                              samples=len(jobs_sorted)),
                         worst_pump_ms=dict(ms=worst_pump[0], t=worst_pump[1]),
                         peak_pump_ms=dict(ms=pump_peak[0], t=pump_peak[1]),
                         lines=async_count, stages=stages_tail, t=round(t, 1))

# BUILD RATE: one row per 60 s window; a serial build is a median gap over 1 s
# with at most one worker busy for five consecutive windows.
build_windows = []
for wnd in sorted(set(window_finishes) | set(window_busy)):
    times = sorted(window_finishes.get(wnd, []))
    gaps_s = [round(b - a, 3) for a, b in zip(times, times[1:])]
    median = round(statistics.median(gaps_s), 3) if gaps_s else None
    busy = window_busy.get(wnd)
    serial = median is not None and median > 1.0 and busy is not None and busy <= 1
    build_windows.append(dict(window=wnd, modules=len(times), median=median,
                              busy=busy, serial=serial))
serial_best = run = 0
for bw in build_windows:
    run = run + 1 if bw['serial'] else 0
    serial_best = max(serial_best, run)
findings['build_rate'] = dict(windows=build_windows, serial_build=serial_best >= 5,
                              serial_windows=serial_best, totals=dict(build_totals),
                              largest=[rec for _, _, rec in sorted(build_top, reverse=True)],
                              t=round(t, 1))

findings['parked'] = dict(threads=park_threads, summary=park_summary, t=round(t, 1))
findings['holes'] = [dict(reason=k, shaders=len(v['shaders']), holes=v['count'],
                          t_first=v['t_first'], t_last=v['t_last'], examples=v['examples'])
                     for k, v in sorted(holes.items(), key=lambda kv: -kv[1]['count'])]

failed_programs = None; cache_t = None
for text, tt in cache_lines:
    if 'failed programs' in text:
        failed_programs = sum(kv(text.split('failed programs', 1)[1]).values()); cache_t = tt
findings['store'] = dict(cache_lines=cache_lines, last_store=last_store, lookups=store_lookups,
                         failed_programs=failed_programs, cache_t=cache_t, t=round(t, 1),
                         miss_fields=[dict(field=k, count=v['count'], shaders=len(v['shaders']),
                                           t_first=v['t_first'], t_last=v['t_last'])
                                      for k, v in sorted(store_miss_fields.items(),
                                                         key=lambda kv: -kv[1]['count'])])

findings['rejections'] = dict(skipped_shaders=len(skips), skip_lines=skip_lines,
                              skip_t_first=skip_t_first, skip_t_last=skip_t_last, t=round(t, 1))

registered = used = None
lead = {}
for line in ledger_lines:
    body = line.split('] ', 1)[1] if '] ' in line else line
    if body.startswith('registered:'): registered = kv(body.split(':', 1)[1])
    elif body.startswith('used:'): used = kv(body.split(':', 1)[1])
    m = lead_re.search(line)
    if m and int(m.group(2)):
        lead[m.group(1)] = dict(n=int(m.group(2)), min=int(m.group(3)) if m.group(3) else None)
findings['ledger'] = dict(registered=registered, used=used, lead=lead, t=ledger_t)

stages_text = stages_tail or ''
def stages_dict(tail):
    d = {}
    for name, pat in (('requested', r'requested (\d+)'), ('translating', r'translating (\d+)'),
                      ('translated', r'translated (\d+)'), ('translated_failed', r'translated \d+ \(failed (\d+)\)'),
                      ('module_compiling', r'module-compiling (\d+) of (\d+) jobs'),
                      ('modules_ready', r'modules ready (\d+)'), ('expired', r'expired (\d+)'),
                      ('wanted', r'wanted (\d+)')):
        m = re.search(pat, tail)
        if m: d[name] = int(m.group(1))
    m = re.search(r'module-compiling (\d+) of (\d+) jobs', tail)
    if m: d['module_compiling_total'] = int(m.group(2))
    return d
findings['async']['stages_parsed'] = stages_dict(stages_text)

# ---- verdict: the frame timeline after loading leads, then the ranked numbers -
def build_verdict():
    v = []
    if loading_gap is not None:
        shown = ['#%d @%s s gap %s s' % (no, tt, gap)
                 for tt, no, gap in post_loading_frames[:12]]
        if len(post_loading_frames) > 12:
            shown.append('(+%d more)' % (len(post_loading_frames) - 12))
        v.append('after loading gap %s-%s s (%s s): frames %s' %
                 (loading_gap['start'], loading_gap['end'], loading_gap['seconds'],
                  ', '.join(shown) or 'none'))
    if stall_list:
        top = stall_list[0]
        what = ', '.join('%s (%s) %.1f s shader %s/%s' % (p['phase'], p['state'], p['seconds'], p['hash'], p['stage'])
                         for p in top['phases'][:2]) or top.get('raw', '?')
        v.append('longest stall %s s @%s s waited on %s' % (top['seconds'], top['t_last'], what))
    if gaps:
        g = max(gaps, key=lambda x: x['seconds'])
        v.append('frame gap %s s from %s to %s s (frame %s)' % (g['seconds'], g['start'], g['end'], g['frame']))
    elif last_frame_t is not None:
        v.append('last frame @%s s (%s frames)' % (last_frame_t, frame_count))
    if slowest:
        s = slowest[0][2]
        v.append('slowest shader %s/%s %s ms summed (@%s s), worst phase %s %s ms' %
                 (slowest[0][1], s[3], s[0], s[5], s[2], s[1]))
    rep = findings['shaders']['repeated']
    if rep:
        r = max(rep, key=lambda x: x['count'])
        v.append('shader %s translated %d times @%s' % (r['hash'], r['count'], ','.join(str(x) for x in r['times'][:6])))
    look = findings['store']['lookups']
    if look:
        misses = look.get('distinct-miss')
        if misses is None:
            misses = look.get('miss', look.get('miss-key', look.get('miss-absent', '?')))
        v.append('store lookups %s hit / %s distinct miss @%s s' %
                 (look.get('hit', '?'), misses, look['t']))
    elif findings['store']['last_store']:
        st = findings['store']
        v.append('store misses: %s failed programs at load @%s s, %s failed writes @%s s' %
                 (st['failed_programs'] if st['failed_programs'] is not None else 'no data', st['cache_t'],
                  kv(st['last_store'][0]).get('failed', '?'), st['last_store'][1]))
    mw = findings['queues']['memory_waits']
    if mw:
        w = mw[0]
        v.append('longest memory wait %s s @%s s on %s: compare %s, wanted %s seen %s, %d polls' %
                 (w['seconds'], w['t'], w['address'], w['compare'], w['wanted'], w['seen'], w['polls']))
    return v

try:
    verdict = build_verdict()
except Exception as exc:
    verdict = ['section failed: %s' % exc]

out = dict(log=path, lines=lines, run_seconds=round(t, 1), shaders_seen=len(compiled), by_stage=dict(stages),
           last_phase_histogram=dict(phases.most_common(12)),
           new_shaders_per_10s=dict(sorted(shader_t.items())), flip_waits=flips, flip_wait_timeouts=timedout,
           flip_waits_per_10s=dict(sorted(flip_t.items())), skipped=skips, other_bang_lines=dict(bang.most_common(25)),
           unresolved_stubs_called=dict(unresolved_called), device_lost=device_lost, driver_hang=driver_hang, fatal=fatal,
           parse_errors=parse_errors, verdict=verdict, findings=findings, failure_classes=[])

# ---- sections: one failure cannot hide the others ---------------------------
def section(name, lines):
    print('[%s]' % name)
    if not lines: print('  no data')
    else:
        for line in lines: print('  ' + line)


def safe(name, builder):
    try:
        return builder()
    except Exception as exc:
        return ['section failed: %s' % exc]


def lines_timeline():
    if last_frame_t is None:
        return ['no data']
    sec = ['@%s s: %d frames, last frame @%s s (frame %s); per 10 s: %s' %
           (round(t, 1), frame_count, last_frame_t, last_frame_no, dict(sorted(frame_t.items())))]
    sec.append('gaps >= 3 s: %d; %s' % (len(gaps), '; '.join('from %s to %s s (%s s, frame %s)' %
               (g['start'], g['end'], g['seconds'], g['frame']) for g in gaps) if gaps else 'none'))
    if loading_gap is not None:
        sec.append('first gap >= 20 s (loading): %s-%s s (%s s)' %
                   (loading_gap['start'], loading_gap['end'], loading_gap['seconds']))
        sec.append('frames after it: %s' % (', '.join('#%d @%s s gap %s s' % (no, tt, gap)
                   for tt, no, gap in post_loading_frames) or 'none'))
    return sec


def lines_stalls():
    sec = []
    for g in stall_list:
        ph = ' -> '.join('%s/%s %.1f s shader %s stage %s @%s s' %
                         (p['phase'], p['state'], p['seconds'], p['hash'], p['stage'], p['t']) for p in g['phases'])
        sec.append('gap %s s frame %s @%s-%s s: %s' % (g['seconds'], g['frame'], g['t_first'], g['t_last'],
                                                       ph or g.get('raw', '?')))
    return sec


def lines_queues():
    sec = []
    for q in findings['queues']['suspends']:
        sec.append('@%s s queue %s: longest shader suspension %s s on %s' % (q['t'], q['queue'], q['seconds'], q['hash']))
    for w in findings['queues']['memory_waits']:
        sec.append('@%s s queue %s %s: %s, compare %s (wanted %s, seen %s), %d polls, %s s (%d samples)' %
                   (w['t'], w['queue'], w['kind'], w['address'], w['compare'], w['wanted'], w['seen'],
                    w['polls'], w['seconds'], w['count']))
    return sec


def lines_shaders():
    sec = []
    for o in findings['shaders']['outcomes']:
        sec.append('@%s-%s s %s %s: %d' % (o['t_first'], o['t_last'], o['stage'], o['outcome'], o['count']))
    if slowest:
        sec.append('slowest %d by summed phase ms: %s' % (len(slowest), '; '.join(
            '%s/%s %s ms (@%s-%s s, worst %s %s ms)' % (h, v[3], v[0], v[4], v[5], v[2], v[1]) for _, h, v in slowest)))
    for r in findings['shaders']['repeated']:
        sec.append('@%s s %s translated %d times (emitted only): %s' %
                   (r['times'][0], r['hash'], r['count'], ','.join(str(x) for x in r['times'])))
    b = findings['shaders']['render_built']
    if b['count']:
        sec.append('render-thread builds: %d, %d ms total, %d ms worst @%s s' % (b['count'], b['total_ms'], b['worst_ms'], b['t_worst']))
    return sec


def lines_async():
    if not async_count:
        return []
    return ['@%s s: %d Async lines; peak queued %d @%s s, peak waiting %d @%s s' %
            (round(t, 1), async_count, peak_queued[0], peak_queued[1], peak_waiting[0], peak_waiting[1]),
            'jobs/s while queued: min %s, median %s (%d samples); pump worst %s ms @%s s (max %s ms @%s s)' %
            (jobs_sorted[0] if jobs_sorted else 'no data',
             round(statistics.median(jobs_sorted), 1) if jobs_sorted else 'no data', len(jobs_sorted),
             worst_pump[0], worst_pump[1], pump_peak[0], pump_peak[1]),
            'final stages @%s s: %s' % (round(t, 1), stages_text or 'no data')]


def lines_build_rate():
    br = findings['build_rate']
    if not br['windows'] and not br['totals']:
        return ['no data']
    sec = []
    for bw in findings['build_rate']['windows']:
        sec.append('window %d: %d modules finished, median interval %s s, peak busy %s%s' %
                   (bw['window'], bw['modules'], bw['median'] if bw['median'] is not None else 'no data',
                    bw['busy'] if bw['busy'] is not None else 'no data',
                    ' [serial]' if bw['serial'] else ''))
    sec.append('serial build: %s (%d consecutive windows with median > 1 s and <= 1 worker busy)' %
               (findings['build_rate']['serial_build'], findings['build_rate']['serial_windows']))
    totals = findings['build_rate']['totals']
    sec.append('build ms totals: translate %d emit %d module %d pipeline %d' %
               (totals.get('translate', 0), totals.get('emit', 0),
                totals.get('module', 0), totals.get('pipeline', 0)))
    for rec in findings['build_rate']['largest']:
        sec.append('%s @%s s total %d ms (translate %d, emit %d, module %d, pipeline %d)' %
                   (rec['hash'], rec['t'], rec['total'], rec['translate'], rec['emit'], rec['module'], rec['pipeline']))
    return sec


def lines_fault_cost():
    if not fault_rows:
        return []
    n = len(fault_rows)
    col = lambda i: [r[i] for r in fault_rows]
    mean = lambda v: sum(v) / len(v)
    secs = sum(col(1))
    sec = ['@%s-%s s: %d intervals (%.0f s); rates per second, mean / peak:' % (fault_rows[0][0], fault_rows[-1][0], n, secs)]
    for label, i, unit in (('write faults', 2, '/s'), ('read faults', 3, '/s'), ('pages armed', 4, ' pg/s'),
                           ('pages released', 5, ' pg/s'), ('protect calls', 6, '/s'),
                           ('in protect syscalls', 7, ' ms/s'), ('in fault handler', 8, ' ms/s'),
                           ('in buffer invalidate', 9, ' ms/s'), ('fast skips (no lock)', 10, '/s')):
        sec.append('  %-22s %9.1f / %9.1f%s' % (label, mean(col(i)), max(col(i)), unit))
    waits = ['%s=%.1f' % (k, v / n) for k, v in sorted(fault_waits.items(), key=lambda kv: -kv[1]) if v > 0]
    sec.append('  lock wait ms/s by caller (mean): %s' % (' '.join(waits) or 'none'))
    storm_on = fault_rows[-1][11]
    sec.append('  fault-storm rule: %s; pages marked %d, re-arms skipped %.0f pg/s (mean), pages expired %d' %
               ('on' if storm_on else 'off', sum(col(12)), mean(col(13)), sum(col(14))))
    if fault_blocks:
        sec.append('  top 64K blocks by faults over the run: %s' %
                   ' '.join('%s=%d' % (a.decode(), c) for a, c in fault_blocks.most_common(8)))
    hot = mean(col(8))
    if hot >= 100:
        sec.append('  VERDICT: the handler takes %.0f ms of every second (%.0f%% of one core): see the '
                   'lock-wait split and the top blocks' % (hot, hot / 10))
    return sec


def lines_parked():
    sec = []
    for p in park_threads:
        sec.append('@%s s thread %d parked at %s (pc %s)' % (p['t'], p['id'], p['module'], p['pc']))
    for tt, count in park_summary:
        sec.append('@%s s %d guest threads parked' % (tt, count))
    return sec


def lines_holes():
    return ['@%s-%s s: %d shaders / %d holes: %s' %
            (h['t_first'], h['t_last'], h['shaders'], h['holes'], h['reason'])
            for h in findings['holes']]


def lines_store():
    sec = ['@%s s: %s' % (tt, x) for x, tt in cache_lines]
    if last_store: sec.append('@%s s: %s' % (last_store[1], last_store[0]))
    look = store_lookups
    if look:
        sec.append('@%s s lookups: total=%s hit=%s distinct-miss=%s miss-key=%s miss-absent=%s' %
                   (look['t'], look.get('lookups', '?'), look.get('hit', '?'),
                    look.get('distinct-miss', '?'), look.get('miss-key', '?'), look.get('miss-absent', '?')))
    elif cache_lines or last_store:
        sec.append('@%s s lookups: no data' % round(t, 1))
    for f in findings['store']['miss_fields']:
        sec.append('@%s-%s s miss field %s: %d lines, %d distinct shaders' %
                   (f['t_first'], f['t_last'], f['field'], f['count'], f['shaders']))
    return sec


def lines_rejections():
    sec = []
    if skip_lines:
        sec.append('@%s-%s s: %d "!!! SHADER SKIP" lines, %d shaders' % (skip_t_first, skip_t_last, skip_lines, len(skips)))
    classes = collections.defaultdict(list)
    for h, v in skips.items():
        classes[(v['site'], re.sub(r'(?<!opcode=)0x[0-9a-f]+|\b\d+\b', 'N', v['reason'])[:200])].append(h)
    out['failure_classes'] = [dict(site=k[0], reason=k[1], shaders=len(v), uses=sum(skips[h]['uses'] for h in v), examples=v[:3])
                              for k, v in sorted(classes.items(), key=lambda kv: -len(kv[1]))]
    findings['rejections']['failure_classes'] = out['failure_classes']
    for c in out['failure_classes']:
        sec.append('@%s s: %d shaders / %d uses @%s: %s' %
                   (skips[c['examples'][0]]['t'], c['shaders'], c['uses'], c['site'], c['reason'][:120]))
    return sec


def lines_ledger():
    if not (registered or used or lead):
        return []
    return ['@%s s registered/used: %s' % (ledger_t, ' '.join('%s=%s/%s' % (s, (registered or {}).get(s, '?'), (used or {}).get(s, '?'))
                                                       for s in sorted(set(registered or {}) | set(used or {})))),
            '@%s s lead time min ms: %s' % (ledger_t, ' '.join('%s=%s' % (s, v['min']) for s, v in sorted(lead.items())))]


print("VERDICT:")
if not verdict: print("  no data")
for i, line in enumerate(verdict, 1): print("  %d. %s" % (i, line))
print("FINDINGS:")
for name, builder in (('TIMELINE', lines_timeline), ('STALLS', lines_stalls),
                      ('QUEUES', lines_queues), ('SHADERS', lines_shaders),
                      ('ASYNC', lines_async), ('BUILD RATE', lines_build_rate),
                      ('PARKED', lines_parked), ('FAULT COST', lines_fault_cost), ('HOLES', lines_holes),
                      ('STORE', lines_store), ('REJECTIONS', lines_rejections),
                      ('LEDGER', lines_ledger)):
    section(name, safe(name, builder))


def summary():
    print("log %s: %d lines, ~%s s of wall time" % (path, lines, out['run_seconds']))
    print("shaders seen: %d  by stage: %s" % (len(compiled), dict(stages)))
    print("last finished phase per shader:", dict(phases.most_common(12)))
    print("new shaders per 10 s:", out['new_shaders_per_10s'])
    print("flip waits: %d (timeouts %d); per 10 s: %s" % (flips, timedout, out['flip_waits_per_10s']))
    print("skipped shaders: %d" % len(skips))
    print("failure classes: %d  (one class = one fix; shaders / uses (>=, power-of-two counts) / site / reason)" % len(out['failure_classes']))
    for c in out['failure_classes']:
        print("  %4d sh %8d uses  %s  %s  e.g. %s" % (c['shaders'], c['uses'], c['site'], c['reason'][:150], c['examples'][0]))
    print("other '!!!' lines:")
    [print("  %6d x %s" % (n, k)) for k, n in bang.most_common(25)]
    out['calls'] = {k: dict(count=v[0], first=round(v[1], 1), last=round(v[2], 1)) for k, v in calls.items()}
    if calls:
        libs = collections.Counter()
        for k, v in calls.items(): libs[k.split('::')[0]] += v[0]
        print("system calls traced: %d functions (counts are lower bounds: 1-4, then powers of two)" % len(calls))
        print("  by library:", dict(libs.most_common(40)))
        late = sorted((v[1], k, v[0]) for k, v in calls.items())[-25:]
        print("  last 25 functions to appear (guest s, name, calls):")
        for ft, k, n in late: print("    %7.1f  %s  x%d" % (ft, k, n))
        for lib in ('Pad', 'UserService', 'SystemService', 'SaveData', 'NpManager'):
            fs = {k.split('::')[-1]: v[0] for k, v in calls.items() if k.startswith(lib + '::')}
            print("  %s: %s" % (lib, fs if fs else 'never called'))
    print("unresolved stubs called:", dict(unresolved_called))
    if device_lost:
        print("GPU DEVICE LOST TRIAGE:")
        [print("  " + x) for x in device_lost]
    # Driver (RADV_DEBUG=hang) dump folders: file list, pipeline.log size, the fault line.
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
                    if n == 'vm_fault.log' and os.path.getsize(p) == 0:
                        fault.append("vm_fault.log: empty (no per-process GPUVM fault; kernel devcoredump fault may be stale)")
                    for l in open(p, errors='replace'):
                        if re.search(r'fault|address', l, re.I) and l.strip():
                            fault.append("%s: %s" % (n, l.strip()[:200]))
            out['driver_dumps'].append(dict(path=d, files=names, pipeline_log_bytes=os.path.getsize(pl) if os.path.isfile(pl) else None, fault_lines=fault[:6]))
            print("  %s: %d files: %s" % (d, len(names), ' '.join(names)))
            print("  pipeline.log: %s bytes" % (os.path.getsize(pl) if os.path.isfile(pl) else 'missing'))
            [print("  " + x) for x in fault[:6]]
        if not dumps: print("  no radv_dumps_* folder found")

try:
    summary()
except Exception as exc:
    print("section failed: %s" % exc)
try:
    print("[RUN-ENDER]")
    print("FATAL:" if fatal else "no fatal block")
    [print("  " + x) for x in fatal]
except Exception as exc:
    print("section failed: %s" % exc)
if '--json' in sys.argv:
    try:
        json.dump(out, open(sys.argv[sys.argv.index('--json') + 1], 'w'), indent=1)
    except Exception as exc:
        print("section failed: %s" % exc)
