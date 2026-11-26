#!/usr/bin/env python3
"""Deterministic tests for tools/bringup/triage.py.

Each finding is checked with a small hand-written log; the excerpt test checks a
2,000-line masked cut of a real run. No emulator or run data is needed."""
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
        m = re.match(r'^\[([A-Z-]+)\]$', line)
        if m:
            current = m.group(1)
            result[current] = []
        elif current and line.startswith('  ') and line.strip():
            result[current].append(line[2:])
    return result


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
        self.assertIn('lookups: hit=80 miss=20', '\n'.join(sections(out)['STORE']))

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
        for name in ('TIMELINE', 'STALLS', 'QUEUES', 'SHADERS', 'ASYNC', 'STORE', 'REJECTIONS', 'LEDGER'):
            self.assertEqual(found[name], ['no data'], name)
        self.assertIn('  no data', out.split('VERDICT:\n', 1)[1].split('FINDINGS:', 1)[0])
        self.assertEqual(out.strip().splitlines()[-1], 'no fatal block')

    def test_run_ender_is_last(self):
        out, _ = run_triage('@@T 1\n--- Error ---\nsome error\n')
        self.assertIn('[RUN-ENDER]\nFATAL:\n  --- Error ---', out)
        self.assertEqual(re.findall(r'^\[([A-Z-]+)\]$', out, re.M)[-1], 'RUN-ENDER')

    def test_excerpt(self):
        out, data = run_triage_path(EXCERPT, os.path.join(tempfile.mkdtemp(), 'triage.json'))
        with open(EXCERPT, 'rb') as f:
            self.assertEqual(sum(1 for _ in f), 2000)
        for name in ('timeline', 'stalls', 'queues', 'shaders', 'async', 'store', 'rejections', 'ledger'):
            self.assertTrue(data['findings'][name], name)
        self.assertTrue(data['verdict'])
        self.assertEqual(len(data['verdict']), 6)
        for name in ('TIMELINE', 'STALLS', 'QUEUES', 'SHADERS', 'ASYNC', 'STORE', 'REJECTIONS', 'LEDGER'):
            self.assertTrue(sections(out)[name], name)
        with open(EXCERPT, 'rb') as f:
            for line in f:
                for h in re.findall(rb'0x[0-9a-fA-F]{16}|(?<![\w])[0-9a-f]{16}(?![\w])', line):
                    h = h.lower()
                    self.assertTrue(h.startswith(b'0xaaaa') or
                                    re.fullmatch(rb'0?x?0{6}[0-9a-f]{10}', h),
                                    'unmasked value: %r' % h)


if __name__ == '__main__':
    unittest.main()
