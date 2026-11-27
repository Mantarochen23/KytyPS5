#!/usr/bin/env python3
"""Deterministic tests for tools/bringup/triage.py.

Each finding is checked with a small hand-written log; the excerpt test checks a
2,000-line masked cut of a real run. No emulator or run data is needed. All
shader ids below are 0xaaaa... pseudonyms, never real hashes.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TRIAGE = os.path.normpath(os.path.join(HERE, '..', 'triage.py'))
EXCERPT = os.path.join(HERE, 'data', 'excerpt.log')


def run_triage(text):
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, 'snippet.log')
        out = os.path.join(d, 'triage.json')
        with open(log, 'w') as f:
            f.write(text)
        return run_triage_path(log, out)


def run_triage_path(log, out=None):
    args = [sys.executable, TRIAGE, log]
    if out:
        args += ['--json', out]
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if p.returncode or not p.stdout.startswith('VERDICT:'):
        raise AssertionError('triage failed: %d %s' % (p.returncode, p.stderr))
    data = None
    if out:
        with open(out) as f:
            data = json.load(f)
    return p.stdout, data


def sections(text):
    result = {}
    current = None
    for line in text.splitlines():
        m = re.match(r'^\[([A-Z][A-Z -]*)\]$', line)
        if m:
            current = m.group(1)
            result[current] = []
        elif current and line.startswith('  ') and line.strip():
            result[current].append(line[2:])
    return result


ALL_SECTIONS = ('TIMELINE', 'STALLS', 'QUEUES', 'SHADERS', 'ASYNC',
                'BUILD RATE', 'PARKED', 'FAULT COST', 'HOLES', 'STORE', 'REJECTIONS', 'LEDGER')


class TriageTests(unittest.TestCase):
    def test_timeline(self):
        text = ('@@T 1\n\t flip_arg             = 1\n@@T 2\n\t flip_arg             = 2\n'
                '@@T 6\n\t flip_arg             = 3\n@@T 7\n\t flip_arg             = 4\n'
                '@@T 12\n\t flip_arg             = 5\n')
        out, data = run_triage(text)
        f = data['findings']['timeline']
        self.assertEqual(f['frames'], 5)
        self.assertEqual(f['frames_per_10s'], {'0': 4, '10': 1})
        self.assertEqual(f['last_frame_t'], 12.0)
        self.assertEqual([(g['start'], g['seconds']) for g in f['gaps']], [(2.0, 4.0), (7.0, 5.0)])
        self.assertTrue(sections(out)['TIMELINE'][0].startswith('@12.0 s: 5 frames, last frame @12.0 s'))
        self.assertIsNone(f['loading_gap'])           # no 20 s gap here

    def test_verdict_leads_with_frames_after_loading_gap(self):
        text = ('@@T 1\n\t flip_arg = 1\n@@T 2\n\t flip_arg = 2\n'
                '@@T 30\n\t flip_arg = 3\n@@T 31\n\t flip_arg = 4\n')
        out, data = run_triage(text)
        f = data['findings']['timeline']
        self.assertEqual(f['loading_gap'], dict(start=2.0, end=30.0, seconds=28.0, frame=3))
        self.assertEqual([(a, b, c) for a, b, c in
                          ((r['t'], r['frame'], r['gap']) for r in f['frames_after_loading'])],
                         [(30.0, 3, 28.0), (31.0, 4, 1.0)])
        first = data['verdict'][0]
        self.assertTrue(first.startswith('after loading gap 2.0-30.0 s (28.0 s):'))
        self.assertIn('#3 @30.0 s gap 28.0 s', first)
        self.assertIn('#4 @31.0 s gap 1.0 s', first)
        self.assertTrue(out.split('VERDICT:\n', 1)[1].splitlines()[0].endswith(first))

    def test_stalls(self):
        text = ("@@T 20\n[stall] no frame for 5.0 s (frame 9): render thread in 'cp-blocked' "
                "state=a queue waits on guest memory or a flip for 0.0 s; shader hash = 0x0000000000000000 "
                "stage=-; last PM4 opcode 0x15, 1 packets; shader worker jobs pending 1\n"
                "@@T 25\n[stall] no frame for 10.0 s (frame 9): render thread in 'dispatch-shader' "
                "state=lookup for 2.5 s; shader hash = 0xaaaa000000000001 stage=cs; last PM4 opcode 0x16, "
                "2 packets; shader worker jobs pending 1\n")
        out, data = run_triage(text)
        st = data['findings']['stalls']
        self.assertEqual(len(st), 2)
        self.assertEqual([p['phase'] for p in st[0]['phases']], ['dispatch-shader'])
        self.assertEqual(st[0]['phases'][0]['seconds'], 2.5)
        self.assertEqual(st[0]['phases'][0]['hash'], '0xaaaa000000000001')
        self.assertEqual(st[0]['phases'][0]['stage'], 'cs')
        self.assertEqual(st[0]['t_last'], 25.0)
        self.assertIn('dispatch-shader', sections(out)['STALLS'][0])

    def test_queues(self):
        wait = ('[wait] queue 0x20 (compute): wait_reg_mem32 on memory 0x0000001000001234 '
                'mask 0xffffffff compare (value & mask) >= 0x7, value at last poll 0x6; '
                'packet at 0x000000104b000080 + 4 dw; polled %d times in %s s\n')
        text = ('@@T 30\n[wait] queue 0 (draw): suspended for shader 0xaaaa000000000001 for 2.5 s\n'
                + wait % (10, '4.9') + '@@T 31\n' + wait % (20, '5.1'))
        out, data = run_triage(text)
        q = data['findings']['queues']
        self.assertEqual(q['suspends'], [{'queue': '0', 'seconds': 2.5, 'hash': '0xaaaa000000000001', 't': 30.0}])
        self.assertEqual(q['memory_wait_lines'], 1)
        self.assertEqual(len(q['memory_waits']), 1)
        w = q['memory_waits'][0]
        self.assertEqual((w['address'], w['wanted'], w['seen'], w['polls'], w['seconds']),
                         ('0x0000001000001234', '0x7', '0x6', 20, 5.1))
        self.assertIn('wanted 0x7, seen 0x6', sections(out)['QUEUES'][1])

    def test_shaders(self):
        h1, h2 = '0xaaaa000000000001', '0xaaaa000000000002'
        text = ('@@T 40\n'
                'ShaderRecompiler CS phase end: hash=%s CFG BuildGraph blocks=1 loops=0 back_edges=0 elapsed_ms=1\n'
                'ShaderRecompiler CS phase end: hash=%s IR TranslateProgram blocks=2 elapsed_ms=9\n'
                'ShaderRecompiler CS phase end: hash=%s CFG BuildGraph blocks=1 loops=0 back_edges=0 elapsed_ms=3\n'
                'ShaderRecompiler CS translation end: hash=%s outcome=emitted\n'
                'ShaderRecompiler CS translation end: hash=%s outcome=emitted\n'
                'ShaderRecompiler PS translation end: hash=%s outcome=served from store\n'
                'Async STALL: the render thread built 100 ms in 2 s (worst 70 ms): compute 100 ms (1 shaders, 0 pipelines)\n'
                % (h1, h1, h2, h1, h1, h2))
        out, data = run_triage(text)
        s = data['findings']['shaders']
        self.assertEqual([(o['stage'], o['outcome'], o['count']) for o in s['outcomes']],
                         [('CS', 'emitted', 2), ('PS', 'served from store', 1)])
        self.assertEqual((s['slowest'][0]['hash'], s['slowest'][0]['total_ms'],
                          s['slowest'][0]['worst_phase'], s['slowest'][0]['worst_ms']),
                         (h1, 10, 'IR TranslateProgram', 9))
        self.assertEqual((s['repeated'][0]['hash'], s['repeated'][0]['count'], s['repeated'][0]['times']),
                         (h1, 2, [40.0, 40.0]))
        self.assertEqual((s['render_built']['total_ms'], s['render_built']['worst_ms']), (100, 70))
        self.assertIn('render-thread builds: 1, 100 ms total, 70 ms worst', '\n'.join(sections(out)['SHADERS']))

    def test_translated_times_counts_emitted_only(self):
        h = '0xaaaa000000000001'
        text = ('@@T 1\nShaderRecompiler CS translation end: hash=%s outcome=emitted\n'
                '@@T 2\nShaderRecompiler CS translation end: hash=%s outcome=served from store\n'
                '@@T 3\nShaderRecompiler CS translation end: hash=%s outcome=served from store\n'
                '@@T 4\nShaderRecompiler CS translation end: hash=%s outcome=emitted\n') % (h, h, h, h)
        out, data = run_triage(text)
        s = data['findings']['shaders']
        self.assertEqual([(o['outcome'], o['count']) for o in s['outcomes']],
                         [('emitted', 2), ('served from store', 2)])
        self.assertEqual(len(s['repeated']), 1)
        self.assertEqual((s['repeated'][0]['count'], s['repeated'][0]['times']), (2, [1.0, 4.0]))
        self.assertIn('%s translated 2 times (emitted only)' % h, '\n'.join(sections(out)['SHADERS']))

    def test_async(self):
        tail = ('stages: requested {req}, translating 0, translated 2 (failed 0), '
                'module-compiling 0 of 1 jobs, modules ready 1 (speculative 0, lost 0, failed 0), '
                'pipeline-compiling 0 of 0 jobs, pipelines ready 0 (failed 0), expired 0, wanted 0')
        def line(q, jobs, worst, req):
            return ('Async: 1 translating, 0 compiling, 2 waiting (oldest 0 s), 0 pipelines, '
                    '%d queued, 0 skipped/2s, 0/0 redrawn; workers 1/1 busy, %s jobs/s, '
                    'pump 1.0 ms (worst %s ms, 1 applied, 0 backlog); ' % (q, jobs, worst)) + tail.format(req=req)
        text = ('@@T 50\n' + line(3, '10.0', '0.5', 5) + '\n@@T 51\n' + line(0, '20.0', '9.0', 7)
                + '\n@@T 52\n' + line(1, '30.0', '3.0', 9) + '\n')
        out, data = run_triage(text)
        a = data['findings']['async']
        self.assertEqual((a['peak_queued']['count'], a['peak_queued']['t']), (3, 50.0))
        self.assertEqual(a['jobs_per_second']['min'], 10.0)
        self.assertEqual(a['jobs_per_second']['median'], 20.0)
        self.assertEqual(a['jobs_per_second']['samples'], 2)
        self.assertEqual((a['worst_pump_ms']['ms'], a['worst_pump_ms']['t']), (9.0, 51.0))
        self.assertEqual(a['stages_parsed']['requested'], 9)
        self.assertEqual(a['stages_parsed']['module_compiling_total'], 1)
        self.assertIn('final stages @52.0 s:', '\n'.join(sections(out)['ASYNC']))

    def _busy_line(self):
        return ('Async: 0 translating, 0 compiling, 0 waiting (oldest 0 s), 0 pipelines, '
                '0 queued, 0 skipped/2s, 0/0 redrawn; workers 1/1 busy, 1.0 jobs/s, '
                'pump 0.0 ms (worst 0.0 ms, 0 applied, 0 backlog); stages: requested 0, '
                'translating 0, translated 0 (failed 0), module-compiling 0 of 0 jobs, '
                'modules ready 0 (speculative 0, lost 0, failed 0), pipeline-compiling 0 of 0 '
                'jobs, pipelines ready 0 (failed 0), expired 0, wanted 0')

    def test_build_rate_serial_flag(self):
        parts = []
        for w in range(5):
            base = w * 60
            parts += ['@@T %d' % (base + 1), self._busy_line(),
                      '@@T %d' % (base + 2),
                      'ShaderRecompiler CS translation end: hash=0xaaaa000000000001 outcome=emitted',
                      '@@T %d' % (base + 4),
                      'ShaderRecompiler CS translation end: hash=0xaaaa000000000002 outcome=emitted']
        out, data = run_triage('\n'.join(parts) + '\n')
        br = data['findings']['build_rate']
        self.assertTrue(br['serial_build'])
        self.assertEqual(br['serial_windows'], 5)
        self.assertEqual(len(br['windows']), 5)
        self.assertEqual([w['modules'] for w in br['windows']], [2] * 5)
        self.assertEqual([w['median'] for w in br['windows']], [2.0] * 5)
        self.assertEqual([w['busy'] for w in br['windows']], [1] * 5)
        self.assertIn('serial build: True', '\n'.join(sections(out)['BUILD RATE']))

    def test_build_rate_parallel_is_not_serial(self):
        parts = []
        for w in range(5):
            base = w * 60
            parts += ['@@T %d' % (base + 1), self._busy_line().replace('workers 1/1', 'workers 8/12'),
                      '@@T %d' % (base + 2),
                      'ShaderRecompiler CS translation end: hash=0xaaaa000000000001 outcome=emitted',
                      '@@T %d' % (base + 4),
                      'ShaderRecompiler CS translation end: hash=0xaaaa000000000002 outcome=emitted']
        _, data = run_triage('\n'.join(parts) + '\n')
        br = data['findings']['build_rate']
        self.assertFalse(br['serial_build'])
        self.assertEqual(br['serial_windows'], 0)

    def test_build_lines_totals_and_largest(self):
        text = ('@@T 5\n'
                '[build] cs hash=0xaaaa000000000003 words=437492 translate=31ms plan=2ms emit=90ms '
                'module=410ms pipeline=3320ms queue-wait=12ms total=3865ms where=worker reason=dispatch\n'
                '[build] cs hash=0xaaaa000000000004 words=10 translate=1ms plan=0ms emit=2ms '
                'module=3ms pipeline=4ms queue-wait=0ms total=10ms where=worker reason=dispatch\n')
        out, data = run_triage(text)
        br = data['findings']['build_rate']
        self.assertEqual(br['totals'], dict(translate=32, emit=92, module=413, pipeline=3324))
        self.assertEqual([(r['hash'], r['total']) for r in br['largest']],
                         [('0xaaaa000000000003', 3865), ('0xaaaa000000000004', 10)])
        joined = '\n'.join(sections(out)['BUILD RATE'])
        self.assertIn('translate 32 emit 92 module 413 pipeline 3324', joined)
        self.assertIn('0xaaaa000000000003 @5.0 s total 3865 ms', joined)

    def test_parked(self):
        text = ('@@T 5\n[park] thread \'\' (id 16) parked for good: pc=0x0000000000001234 '
                '= mod+0x1234 access=2 address=0x0000000000000000\n'
                '@@T 6\n[park] 1 guest threads parked\n')
        out, data = run_triage(text)
        p = data['findings']['parked']
        self.assertEqual(len(p['threads']), 1)
        self.assertEqual((p['threads'][0]['id'], p['threads'][0]['module'], p['threads'][0]['t']),
                         (16, 'mod+0x1234', 5.0))
        self.assertEqual(p['summary'], [[6.0, 1]])
        joined = '\n'.join(sections(out)['PARKED'])
        self.assertIn('thread 16 parked at mod+0x1234', joined)
        self.assertIn('1 guest threads parked', joined)

    def test_holes(self):
        text = ('@@T 10\n'
                '!!! SHADER DISPATCH SKIP (resource materialization failed) shader=0xaaaa000000000003; '
                'rendering is incomplete\n'
                '!!! SHADER DISPATCH SKIP (resource materialization failed) shader=0xaaaa000000000004; '
                'rendering is incomplete\n'
                'shader resource specialization failed: atomic image descriptor 12 uses unsupported format 5\n'
                'shader resource specialization failed: atomic image descriptor 13 uses unsupported format 6\n')
        out, data = run_triage(text)
        holes = data['findings']['holes']
        by_reason = {h['reason']: h for h in holes}
        self.assertEqual(by_reason['resource materialization failed; rendering is incomplete']['shaders'], 2)
        self.assertEqual(by_reason['resource materialization failed; rendering is incomplete']['holes'], 2)
        norm_key = 'shader resource specialization failed: atomic image descriptor N uses unsupported format N'
        self.assertEqual(by_reason[norm_key]['holes'], 2)
        joined = '\n'.join(sections(out)['HOLES'])
        self.assertIn('descriptor N uses unsupported format N', joined)

    def test_store(self):
        text = ('@@T 60\n'
                'Shader cache: loaded per stage: programs with modules vs=1 ps=2 cs=3 gs=0 ls=0 hs=0 tes=0; '
                'failed programs vs=0 ps=1 cs=0 gs=0 ls=0 hs=0 tes=0\n'
                'Shader cache: loaded 6 programs, 7 modules, 8 pipelines in 9 ms\n'
                '[store] lookups=100 hit=80 miss=20\n'
                '@@T 61\n[store] written=1 pending=0 failed=0\n')
        out, data = run_triage(text)
        s = data['findings']['store']
        self.assertEqual(len(s['cache_lines']), 2)
        self.assertEqual(s['failed_programs'], 1)
        self.assertEqual(s['last_store'], ['[store] written=1 pending=0 failed=0', 61.0])
        self.assertEqual((s['lookups']['hit'], s['lookups']['miss'], s['lookups']['t']), (80, 20, 60.0))
        self.assertIn('lookups: total=100 hit=80 distinct-miss=? miss-key=? miss-absent=?',
                      '\n'.join(sections(out)['STORE']))

    def test_store_lookups_distinct_miss_and_miss_groups(self):
        text = ('@@T 60\n'
                '[store] lookups=6126448 hit=5198309 miss-key=798238 miss-absent=129815 distinct-miss=1381\n'
                '@@T 61\n[store] miss #4 key stage=PS hash=0xaaaa000000000001: nearest module '
                '0xaaaa000000000002 of 3: samplers count 7 vs 10 (+2 more)\n'
                '@@T 62\n[store] miss #5 key stage=PS hash=0xaaaa000000000005: nearest module '
                '0xaaaa000000000002 of 3: samplers count 8 vs 11 (+2 more)\n'
                '@@T 63\n[store] written=1409 pending=0 failed=0 flushed=1409\n')
        out, data = run_triage(text)
        s = data['findings']['store']
        self.assertEqual((s['lookups']['lookups'], s['lookups']['hit'], s['lookups']['distinct-miss']),
                         (6126448, 5198309, 1381))
        self.assertEqual((s['lookups']['miss-key'], s['lookups']['miss-absent']), (798238, 129815))
        self.assertEqual(len(s['miss_fields']), 1)
        self.assertEqual((s['miss_fields'][0]['field'], s['miss_fields'][0]['count'],
                          s['miss_fields'][0]['shaders']), ('samplers count', 2, 2))
        joined = '\n'.join(sections(out)['STORE'])
        self.assertIn('distinct-miss=1381', joined)
        self.assertIn('miss field samplers count: 2 lines, 2 distinct shaders', joined)
        self.assertIn('distinct miss', data['verdict'][0])

    def test_store_without_lookups(self):
        out, data = run_triage('@@T 1\n[store] written=0 pending=0 failed=0\n')
        self.assertIsNone(data['findings']['store']['lookups'])
        self.assertIn('lookups: no data', '\n'.join(sections(out)['STORE']))

    def test_rejections(self):
        text = ('@@T 70\n'
                '!!! SHADER SKIP (keep-going) shader = 0xaaaa000000000001 stage=cs at site:12: reason one\n'
                '!!! SHADER SKIP USES shader = 0xaaaa000000000001 count=1\n'
                '!!! SHADER SKIP USES shader = 0xaaaa000000000001 count=2\n')
        out, data = run_triage(text)
        r = data['findings']['rejections']
        self.assertEqual((r['skip_lines'], r['skipped_shaders']), (3, 1))
        self.assertEqual(len(data['failure_classes']), 1)
        self.assertEqual(data['failure_classes'][0]['uses'], 2)
        self.assertIn('reason one', '\n'.join(sections(out)['REJECTIONS']))

    def test_ledger(self):
        text = ('@@T 80\n'
                '[shader-reg] === shader registration ledger ===\n'
                '[shader-reg] registered:                cs=10 ps=20 gs=0 hs=0 fs=0 ?=0\n'
                '[shader-reg] used:                      cs=3 ps=4 gs=0 hs=0 fs=0 ?=0\n'
                '[shader-reg] lead time ms (use - registration) cs: n=3 min=100 median=200 p90=300 max=400\n'
                '[shader-reg] lead time ms (use - registration) ps: n=0\n'
                '[shader-reg] === end shader registration ledger ===\n')
        out, data = run_triage(text)
        led = data['findings']['ledger']
        self.assertEqual((led['registered']['cs'], led['used']['cs']), (10, 3))
        self.assertEqual((led['registered']['ps'], led['used']['ps']), (20, 4))
        self.assertEqual(led['lead']['cs']['min'], 100)
        self.assertNotIn('ps', led['lead'])
        self.assertIn('cs=10/3', '\n'.join(sections(out)['LEDGER']))

    def test_absent_sections_say_no_data(self):
        out, data = run_triage('@@T 1\nnothing interesting here\n')
        found = sections(out)
        for name in ALL_SECTIONS:
            self.assertEqual(found[name], ['no data'], name)
        self.assertIn('  no data', out.split('VERDICT:\n', 1)[1].split('FINDINGS:', 1)[0])
        self.assertEqual(out.strip().splitlines()[-1], 'no fatal block')
        self.assertEqual(data['parse_errors'], 0)

    def test_one_section_failure_does_not_hide_the_rest(self):
        with open(TRIAGE) as f:
            source = f.read()
        broken = source.replace('def lines_parked():\n',
                                "def lines_parked():\n    raise ValueError('boom')\n", 1)
        self.assertNotEqual(broken, source)
        with tempfile.TemporaryDirectory() as d:
            script = os.path.join(d, 'broken_triage.py')
            log = os.path.join(d, 'snippet.log')
            with open(script, 'w') as f:
                f.write(broken)
            with open(log, 'w') as f:
                f.write('@@T 1\n[park] 2 guest threads parked\n\t flip_arg = 1\n'
                        '[store] written=0 pending=0 failed=0\n')
            p = subprocess.run([sys.executable, script, log], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, universal_newlines=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            found = sections(p.stdout)
            self.assertEqual(found['PARKED'], ['section failed: boom'])
            self.assertTrue(found['STORE'])
            self.assertTrue(found['TIMELINE'])
            self.assertIn('VERDICT:', p.stdout)

    def test_garbage_lines_never_crash(self):
        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, 'snippet.log')
            with open(log, 'wb') as f:
                f.write(b'@@T 1\n\xff\xfe random \x00 bytes\n[store] written=x pending=? failed=0\n'
                        b'!!! \x00\x01 ???\n\t flip_arg = 1\n')
            out, data = run_triage_path(log, os.path.join(d, 'triage.json'))
            found = sections(out)
            for name in ALL_SECTIONS:
                self.assertTrue(found[name], name)
            self.assertTrue(data['verdict'])

    def test_run_ender_is_last(self):
        out, _ = run_triage('@@T 1\n--- Error ---\nsome error\n')
        self.assertIn('[RUN-ENDER]\nFATAL:\n  --- Error ---', out)
        self.assertEqual(re.findall(r'^\[([A-Z][A-Z -]*)\]$', out, re.M)[-1], 'RUN-ENDER')

    def test_excerpt(self):
        out, data = run_triage_path(EXCERPT, os.path.join(tempfile.mkdtemp(), 'triage.json'))
        with open(EXCERPT, 'rb') as f:
            self.assertEqual(sum(1 for _ in f), 2000)
        for name in ('timeline', 'stalls', 'queues', 'shaders', 'async', 'build_rate',
                     'parked', 'holes', 'store', 'rejections', 'ledger'):
            self.assertIn(name, data['findings'], name)
        self.assertTrue(data['verdict'])
        self.assertEqual(len(data['verdict']), 6)
        for name in ALL_SECTIONS:
            self.assertTrue(sections(out)[name], name)
        with open(EXCERPT, 'rb') as f:
            for line in f:
                for h in re.findall(rb'0x[0-9a-fA-F]{16}|(?<![\w])[0-9a-f]{16}(?![\w])', line):
                    h = h.lower()
                    self.assertTrue(h.startswith(b'0xaaaa') or
                                    re.fullmatch(rb'0?x?0{6}[0-9a-f]{10}', h),
                                    'unmasked value: %r' % h)


class FaultCostTest(unittest.TestCase):
    LOG = (
        "@@T 10.0\n"
        "[fault] 5.0s write=2000/s read=10/s armed=2500 pg/s released=2000 pg/s protect-calls=2400/s "
        "protect=150.0 ms/s handler=300.0 ms/s invalidate=120.0 ms/s fast-skip=40/s storm(on=3) "
        "marked=5 kept=100 pg/s expired=1\n"
        "[fault] lock-wait ms/s (waits/s): other=0.0(0) fault=90.5(800) upload=30.0(20) download=0.0(0) "
        "mark=0.0(0) unmark=0.0(0) texture=1.5(3)\n"
        "[fault] top 64K blocks: 0xaaaa0000=900 0xaaaa10000=300\n"
        "@@T 15.0\n"
        "[fault] 5.0s write=1000/s read=0/s armed=1200 pg/s released=1000 pg/s protect-calls=1100/s "
        "protect=50.0 ms/s handler=100.0 ms/s invalidate=60.0 ms/s fast-skip=10/s storm(on=3) "
        "marked=0 kept=50 pg/s expired=0\n"
        "[fault] lock-wait ms/s (waits/s): other=0.0(0) fault=10.0(100) upload=0.0(0) download=0.0(0) "
        "mark=0.0(0) unmark=0.0(0) texture=0.0(0)\n"
        "[fault] top 64K blocks: 0xaaaa0000=500\n")

    def test_section(self):
        text, data = run_triage(self.LOG)
        body = sections(text)['FAULT COST']
        joined = '\n'.join(body)
        self.assertIn('2 intervals', joined)
        self.assertIn('write faults', joined)
        self.assertIn('1500.0 /    2000.0/s', joined)
        self.assertIn('fault=50.2', joined)
        self.assertIn('0xaaaa0000=1400', joined)
        self.assertIn('rule: on', joined)
        self.assertIn('VERDICT', joined)

    def test_absent(self):
        text, _ = run_triage("@@T 1.0\n")
        self.assertEqual(sections(text)['FAULT COST'], ['no data'])


class FrameCostArchTest(unittest.TestCase):
    def test_frame_summary_and_arch_lines_are_captured(self):
        log = (
            "@@T 10.0\n"
            "[frame] seconds=5.00 frames=4 draws=100.00 dispatches=200.00 rebind=10.00 "
            "rebind_fast=9.00 entries=20.00 rebind_skipped=18.00 rebind_resolved=2.00 "
            "resolve=2.00 find_image=50.00 find_hits=50.00 materialize=300.00 "
            "materialize_hits=299.00 materialize_misses=1.00 find_module=300.00 "
            "module_hits=300.00 module_misses=0.00 ms/frame rebind=1.000 resolve=0.500 "
            "find_image=1.000 materialize=2.000 find_module=3.000 wait_gpu=5.000 "
            "wait_guest=0.100 wait_lock=0.100 idle=0.000 other=12.300 total=25.000 "
            "ns/call materialize_hit=500 materialize_miss=5000 module_hit=1000 module_miss=0\n"
            "[frame] arch wait_gpu_bg=15.000 wait_ms(rb=3.000 stream=0.000 unmap=1.000 "
            "gds=0.000 fault=0.000 gc=0.000 rt=0.000 swap=0.000 prio=1.000 other=0.000) "
            "wait_n(rb=2.00 stream=0.00 unmap=0.50 gds=0.00 fault=0.00 gc=0.00 rt=0.00 "
            "swap=0.00 prio=1.00 other=0.00) submits=6.00(slice=4.00 flip=1.00 susp=0.50 "
            "sync=0.50 wb=0.00 fdraw=0.00 pres=0.00 other=0.00) calls_draw(dir=50.00 "
            "ind_gpu=2.00/20.00 ind_cpu=1.00/48.00 ind_zero=0.00 coal=5.00 quad_ex=0.00 "
            "skip=3.00) calls_disp(dir=20.00 ind=180.00 ind_tdim=0.00 guard=180.00 "
            "oa_ex=0.00 skip=0.00 bvh=4.00) barriers_all=500.00 draw_ns(n=2 prog=1000 "
            "rt=400 pipe=300 bind=2000 commit=1200 rec=800) disp_ns(n=3 prog=900 pipe=250 "
            "bind=1800 guard=1400 commit=1100 rec=1600)\n"
        )
        text, data = run_triage(log)
        self.assertEqual(data['parse_errors'], 0)
        self.assertEqual(len(data['findings']['frame_cost']), 2)
        fc = '\n'.join(sections(text)['FRAME COST'])
        self.assertIn('[frame] seconds=5.00', fc)
        self.assertIn('[frame] arch wait_gpu_bg=15.000', fc)
        self.assertIn('draw_ns(n=2', fc)
        self.assertIn('disp_ns(n=3', fc)


if __name__ == '__main__':
    unittest.main()
