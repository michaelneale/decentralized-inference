import copy
from contextlib import chdir
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('canary', ROOT/'scripts/runtime-seed-canary.py')
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


def cache():
    return {'id': C.CACHE_ID, 'key': C.KEY, 'version': C.VERSION,
            'ref': 'refs/heads/main', 'size_in_bytes': C.CACHE_SIZE}


def representative_stats():
    # Pinned v0.16.0 ServerStats; duration/location fields are deliberately omitted
    # from retained evidence. adv_counts uses compiler-qualified language labels.
    return {'stats': {
        'compile_requests': 10, 'requests_unsupported_compiler': 0,
        'requests_not_compile': 0, 'requests_not_cacheable': 0, 'requests_executed': 10,
        'cache_timeouts': 0, 'cache_read_errors': 0, 'non_cacheable_compilations': 0,
        'forced_recaches': 0, 'cache_write_errors': 0, 'cache_writes': 8,
        'compilations': 8, 'compile_fails': 0, 'dist_errors': 0,
        'cache_hits': {'counts': {'Rust': 1, 'C/C++': 1}, 'adv_counts': {'C/C++ [gcc]': 1, 'Rust': 1}},
        'cache_misses': {'counts': {'C/C++': 8}, 'adv_counts': {'C/C++ [gcc]': 8}},
        'cache_errors': {'counts': {}, 'adv_counts': {}}, 'not_cached': {},
    }}


class RuntimeSeedCanaryTests(unittest.TestCase):
    def test_preflight_checks_container_temp_and_retains_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            container = root/'container'
            cache_dir = container/'mesh-llm-sccache'
            cache_dir.mkdir(parents=True)
            (cache_dir/'unexpected-cache').write_text('existing')
            evidence = root/'evidence'
            environment = {
                'CANARY_KEY': C.KEY, 'GITHUB_EVENT_NAME': 'workflow_dispatch',
                'RUNNER_ENVIRONMENT': 'github-hosted', 'RUNNER_ARCH': 'X64',
                'LLAMA_STAGE_BUILD_DIR': C.BUILD_DIR, 'RUSTC_WRAPPER': 'sccache',
                'MESH_LLM_REQUIRE_SCCACHE': '1', 'CARGO_INCREMENTAL': '0',
                'CACHE_NAMESPACE': 'mesh-llm', 'SCCACHE_GHA_ENABLED': 'false',
                'SCCACHE_MULTILEVEL_CHAIN': 'disk', 'SCCACHE_CACHE_SIZE': '2G',
                'LLAMA_STAGE_BACKEND': 'cpu', 'RUNNER_TEMP': str(container),
                'SCCACHE_DIR': str(root/'host-temp/mesh-llm-sccache'),
            }
            with chdir(root), patch.dict(C.os.environ, environment, clear=True), patch.object(
                C.sys, 'argv', ['canary', 'preflight', str(evidence)]
            ), patch.object(C.subprocess, 'check_output') as command:
                self.assertEqual(C.main(), 1)
                command.assert_not_called()
            failure = C.read(evidence/'preflight-failure.json')
            self.assertEqual(failure['reason'], 'initial compiler cache is not empty')
            self.assertEqual(failure['classification'], 'inconclusive')

    def test_container_checkout_is_trusted_before_preflight(self):
        job = yaml.safe_load((ROOT/'.github/workflows/depot-canary.yml').read_text())['jobs']['runtime_seed']
        steps = job['steps']
        self.assertTrue(steps[0]['uses'].startswith('actions/checkout@'))
        self.assertEqual(steps[1], {'name': 'Trust checkout directory',
            'run': 'git config --global --add safe.directory "$GITHUB_WORKSPACE"'})
        preflight = next(step for step in steps if step.get('id') == 'seed_key')
        self.assertNotIn('SCCACHE_DIR', preflight['env'])
        self.assertGreater(steps.index(preflight), 1)

    def test_cache_admission_rejects_missing_shadow_and_wrong_identity(self):
        self.assertEqual(C.cache_identity({'actions_caches': [cache()]}), cache())
        for field, value in [('id', 1), ('version', 'wrong'), ('ref', 'refs/heads/feature'), ('size_in_bytes', 1)]:
            altered = {**cache(), field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                C.cache_identity({'actions_caches': [altered]})
        for entries in ([], [cache(), cache()], [cache(), {**cache(), 'ref': 'refs/heads/feature'}]):
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                C.cache_identity({'actions_caches': entries})

    def test_raw_stats_preserve_languages_and_reject_bad_counters(self):
        raw = representative_stats()
        raw['cache_location'] = 'not retained'
        with patch.object(C.subprocess, 'check_output', return_value=json.dumps(raw)):
            self.assertEqual(C.stats_snapshot(), {'stats': raw['stats']})
        raw['stats']['cache_hits']['counts']['Rust'] = True
        with patch.object(C.subprocess, 'check_output', return_value=json.dumps(raw)), self.assertRaises(ValueError):
            C.stats_snapshot()

    def test_pinned_schema_rejects_missing_and_mistyped_fields(self):
        for field in C.SCALAR_COUNTERS:
            for corruption in ('missing', True, {}):
                with self.subTest(field=field, corruption=corruption):
                    raw = representative_stats()
                    if corruption == 'missing':
                        del raw['stats'][field]
                    else:
                        raw['stats'][field] = corruption
                    with self.assertRaises(ValueError):
                        C.validate_stats(raw)
        for field in ('counts', 'adv_counts'):
            raw = representative_stats()
            raw['stats']['cache_hits'][field] = {'C/C++': True}
            with self.assertRaises(ValueError):
                C.validate_stats(raw)
        raw = representative_stats()
        raw['stats']['not_cached'] = {'reason': {}}
        with self.assertRaises(ValueError):
            C.validate_stats(raw)

    def results(self, root, native_hits=0, warm_time=12):
        for pair in (1, 2, 3):
            for arm in ('cold', 'warm'):
                path = root/f'{pair}-{arm}'
                path.mkdir()
                raw = representative_stats()
                raw['stats']['cache_hits']['counts'] = {'C/C++': native_hits}
                raw['stats']['cache_misses']['counts'] = {'C/C++': 10-native_hits}
                C.save(path/'raw-stats.json', raw)
                C.save(path/'result.json', {'pair': pair, 'arm': arm, 'source': 'a'*40,
                    'image': C.IMAGE, 'epoch': C.EPOCH, 'cache': cache(), 'classification': 'warm-floor-failure' if arm=='warm' and native_hits==0 else 'measured',
                    'total_seconds': 10 if arm=='cold' else warm_time, 'native_hits': native_hits,
                    'native_cacheable_requests': 10, 'action_seconds': 9, 'restore_seconds': 1,
                    'verified': True, 'eligibility_changed': False, 'warm_floor_passed': arm!='warm' or native_hits>0, 'hit_rate':native_hits/10,
                    'language_hits':{'C/C++':native_hits}, 'language_misses':{'C/C++':10-native_hits},
                    'assembler_hits':0, 'assembler_misses':0, 'runner_class':'github-hosted/ubuntu-24.04/X64', 'kernel':'Linux test x86_64', 'run_id': '123', 'run_attempt': '1',
                    'host_cpu': {'model':'test'}, 'runner_image_os':'ubuntu24', 'runner_image_version':'1'})

    def test_negative_native_coverage_and_total_benefit_are_reported_honestly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.results(root)
            C.summarize(root)
            result = C.read(root/'summary.json')
            self.assertFalse(result['native_coverage_observed'])
            self.assertFalse(result['total_benefit_observed'])
            self.assertFalse(result['eligibility_changed'])

    def test_missing_or_mismatched_samples_are_inconclusive(self):
        for mutation in ('missing', 'source', 'floor'):
            with tempfile.TemporaryDirectory() as tmp, self.subTest(mutation=mutation):
                root = Path(tmp)
                self.results(root)
                path = root/'1-warm/result.json'
                value = C.read(path)
                if mutation=='missing':
                    path.unlink()
                else:
                    value['source' if mutation=='source' else 'classification'] = 'different'
                    C.save(path, value)
                with self.assertRaises(ValueError):
                    C.summarize(root)

    def test_canary_uses_real_action_six_fresh_jobs_and_never_saves(self):
        data = yaml.safe_load((ROOT/'.github/workflows/depot-canary.yml').read_text())
        job = data['jobs']['runtime_seed']
        self.assertEqual(job['runs-on'], 'ubuntu-24.04')
        self.assertEqual(job['strategy']['matrix'], {'pair': [1,2,3], 'arm': ['cold','warm']})
        self.assertEqual(job['container']['image'], C.IMAGE)
        action = next(s for s in job['steps'] if s.get('id')=='runtime')
        self.assertEqual(action['uses'], './.github/actions/prepare-native-runtime-input')
        self.assertEqual(action['with'], {'backend':'cpu','target':'x86_64-unknown-linux-gnu','output_dir':'runtime-input','build':'true'})
        self.assertNotIn('actions/cache/save', json.dumps(job))
        self.assertEqual(job['env']['SCCACHE_CACHE_SIZE'], '2G')
        self.assertEqual(job['env']['SCCACHE_GHA_ENABLED'], 'false')
        configure = next(s for s in job['steps'] if s.get('uses')=='./.github/actions/configure-sccache-gha')
        self.assertEqual(set(configure['with'].values()), {'false'})


    def test_complete_floor_failure_still_emits_negative_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.results(root, native_hits=0, warm_time=5)
            path = root/'1-warm/result.json'
            result = C.read(path)
            result.update(classification='warm-floor-failure', warm_floor_passed=False)
            C.save(path, result)
            C.summarize(root)
            summary = C.read(root/'summary.json')
            self.assertEqual(summary['classification'], 'not-qualified')
            self.assertIn('warm-floor-failure', summary['reasons'])
            self.assertEqual(len(summary['paired_seconds_saved']), 3)

    def test_raw_floor_and_workload_cannot_be_overridden(self):
        for mutation in ('floor', 'workload', 'boolean', 'missing'):
            with tempfile.TemporaryDirectory() as tmp, self.subTest(mutation=mutation):
                root = Path(tmp)
                self.results(root)
                path = root/'1-warm'
                raw = C.read(path/'raw-stats.json')
                result = C.read(path/'result.json')
                if mutation == 'floor':
                    raw['stats']['cache_hits']['counts']['C/C++'] = 1
                    raw['stats']['cache_misses']['counts']['C/C++'] = 999
                    result.update(native_hits=1, native_cacheable_requests=1000,
                                  language_hits={'C/C++':1}, language_misses={'C/C++':999},
                                  hit_rate=.001, warm_floor_passed=True, classification='measured')
                elif mutation == 'workload':
                    raw['stats']['cache_misses']['counts']['C/C++'] = 1000
                    result.update(native_cacheable_requests=1000, language_misses={'C/C++':1000})
                elif mutation == 'boolean':
                    raw['stats']['compile_requests'] = True
                else:
                    del raw['stats']['requests_not_cacheable']
                C.save(path/'raw-stats.json', raw)
                C.save(path/'result.json', result)
                with self.assertRaises(ValueError):
                    C.summarize(root)

    def test_optional_host_image_metadata_can_be_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.results(root)
            for path in root.rglob('result.json'):
                value = C.read(path)
                value.update(runner_image_os=None, runner_image_version=None)
                C.save(path, value)
            C.summarize(root)
            self.assertEqual(C.read(root/'summary.json')['classification'], 'not-qualified')

    def test_incomparable_host_and_attempt_do_not_report_benefit(self):
        for field in ('host_cpu', 'runner_image_version', 'run_attempt'):
            with tempfile.TemporaryDirectory() as tmp, self.subTest(field=field):
                root = Path(tmp)
                self.results(root)
                path = root/'1-warm/result.json'
                value = C.read(path)
                value[field] = 'different'
                C.save(path, value)
                with self.assertRaises(ValueError):
                    C.summarize(root)

if __name__ == '__main__':
    unittest.main()
