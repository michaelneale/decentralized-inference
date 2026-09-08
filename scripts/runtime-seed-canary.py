#!/usr/bin/env python3
"""Observe an existing seed against the unchanged CPU package action; never enable it."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import urllib.parse
import urllib.request

IMAGE = 'ghcr.io/mesh-llm/mesh-llm-cuda-runner@sha256:8d93de6ba30173e825a16fdecf011f9c632edc6e1259df7289e491b0a05f829d'
EPOCH = 'mesh-llm-cuda-runner-sha256-' + IMAGE.split(':')[-1]
KEY = 'mesh-llm-sccache-seed-linux-x86_64-img-8d93de6b-epoch-8d93de6b-v2-9522c1c392ee2b2554146347af6629ecfb45ccba2b8b849deb844fe93c53f09f'
VERSION = '6e0f5d9449f86cfe3ca2e00b7bb4d1ce5034f7275d4c6135aa59f5f733af8d8a'
CACHE_ID = 7456497330
CACHE_SIZE = 235465265
BUILD_DIR = '.deps/llama.cpp/build-stage-abi-dynamic-cpu'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    require(path.stat().st_size <= 16 * 1024 * 1024, 'oversized evidence')
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def cache_identity(payload):
    require(isinstance(payload, dict) and isinstance(payload.get('actions_caches'), list), 'invalid cache listing')
    matches = [item for item in payload['actions_caches'] if item.get('key') == KEY]
    require(len(matches) == 1, 'missing seed or branch-shadow cache')
    item = matches[0]
    for key, expected in {'id': CACHE_ID, 'key': KEY, 'version': VERSION,
                          'ref': 'refs/heads/main', 'size_in_bytes': CACHE_SIZE}.items():
        require(item.get(key) == expected, 'seed metadata mismatch: ' + key)
    return item


def fetch_cache():
    url = 'https://api.github.com/repos/Mesh-LLM/mesh-llm/actions/caches?' + urllib.parse.urlencode({'key': KEY, 'per_page': 100})
    request = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + os.environ['GH_TOKEN'], 'Accept': 'application/vnd.github+json'})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    require(payload.get('total_count', 101) <= 100, 'cache listing pagination requires review')
    return cache_identity(payload)


def preflight(directory):
    require(not directory.exists(), 'fresh evidence directory required')
    directory.mkdir()
    require(os.environ.get('CANARY_KEY') == KEY, 'source seed recipe hash changed')
    require(os.environ.get('GITHUB_EVENT_NAME') == 'workflow_dispatch', 'manual canary only')
    require(os.environ.get('RUNNER_ENVIRONMENT') == 'github-hosted' and os.environ.get('RUNNER_ARCH') == 'X64', 'hosted X64 runner required')
    require(os.environ.get('LLAMA_STAGE_BUILD_DIR') == BUILD_DIR, 'runtime path drift')
    for key, value in {'RUSTC_WRAPPER':'sccache', 'MESH_LLM_REQUIRE_SCCACHE':'1', 'CARGO_INCREMENTAL':'0',
                       'CACHE_NAMESPACE':'mesh-llm', 'SCCACHE_GHA_ENABLED':'false', 'SCCACHE_MULTILEVEL_CHAIN':'disk',
                       'SCCACHE_CACHE_SIZE':'2G', 'LLAMA_STAGE_BACKEND':'cpu'}.items():
        require(os.environ.get(key)==value, 'canary environment drift: '+key)
    for key in ('LLAMA_STAGE_USE_SCCACHE','SKIPPY_USE_SCCACHE'):
        require(os.environ.get(key,'1')=='1', 'compiler cache disabled')
    for name in ('MESH_NATIVE_RUNTIME_MODEL_PACKAGE_TOOL', 'CARGO_TARGET_DIR', 'SKIPPY_LLAMA_BUILD_DIR', 'LLAMA_STAGE_FORCE_BUILD', 'SKIPPY_FORCE_LLAMA_BUILD', 'MESH_NATIVE_RUNTIME_GPU_BENCHMARK_TOOL'):
        require(not os.environ.get(name), 'unexpected override: ' + name)
    for name in (BUILD_DIR, 'target', 'runtime-input'):
        require(not Path(name).exists() and not Path(name).is_symlink(), 'existing native/Cargo/package output: ' + name)
    cache = Path(os.environ['SCCACHE_DIR'])
    require(not cache.is_symlink() and (not cache.exists() or not any(cache.iterdir())), 'initial compiler cache is not empty')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    require(source == os.environ['GITHUB_SHA'], 'source differs from dispatch revision')
    for key in ('GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT'):
        require(os.environ.get(key, '').isdigit() and int(os.environ[key]) > 0, 'invalid run metadata: '+key)
    require(os.environ.get('CANARY_PAIR') in ('1', '2', '3') and os.environ.get('CANARY_ARM') in ('cold', 'warm'), 'invalid sample metadata')
    kernel = subprocess.check_output(['uname', '-srm'], text=True).strip()
    require(kernel.startswith('Linux ') and kernel.endswith(' x86_64'), 'invalid kernel identity')
    metadata = fetch_cache()
    with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
        output.write('key='+KEY+'\n')
    cpu = subprocess.check_output(['lscpu', '--json'], text=True)
    cpu_fields = {item['field']: item['data'] for item in json.loads(cpu)['lscpu']}
    host = {key: cpu_fields.get(key) for key in ('Architecture:', 'CPU(s):', 'Model name:', 'Vendor ID:', 'Thread(s) per core:')}
    require(all(host.values()), 'missing host CPU identity')
    save(directory/'context.json', {'schema': 1, 'source': source, 'image': IMAGE, 'epoch': EPOCH,
         'cache': metadata, 'publisher_run': 34230668171, 'publisher_source': 'a6487dd64de7d0df4a2d72145d18ab33be0f0a9c',
         'pair': int(os.environ['CANARY_PAIR']), 'arm': os.environ['CANARY_ARM'],
         'run_id': os.environ['GITHUB_RUN_ID'], 'run_attempt': os.environ['GITHUB_RUN_ATTEMPT'],
         'build_dir': BUILD_DIR, 'initial_outputs_absent': True, 'host_cpu': host,
         'runner_class': 'github-hosted/ubuntu-24.04/X64', 'kernel': kernel,
         'runner_image_os': os.environ.get('ImageOS'), 'runner_image_version': os.environ.get('ImageVersion')})


def restored(directory):
    context = read(directory/'context.json')
    context['restore_seconds'] = time.monotonic() - read(directory/'restore-start.json')['monotonic']
    context['cache_after_restore'] = fetch_cache()
    context['cache_hit'] = os.environ.get('CANARY_CACHE_HIT') == 'true'
    require(context['arm'] == 'cold' or context['cache_hit'], 'warm restore missed: inconclusive')
    require(context['arm'] != 'cold' or not any(Path(os.environ['SCCACHE_DIR']).iterdir()), 'cold cache populated before build')
    save(directory/'context.json', context)


def start(directory):
    subprocess.run(['sccache', '--zero-stats'], check=True)
    initial = stats_snapshot()
    require(initial['stats']['compile_requests']==0, 'counters did not reset')
    save(directory/'initial-stats.json', initial)
    save(directory/'start.json', {'monotonic': time.monotonic()})


def restore_start(directory):
    save(directory/'restore-start.json', {'monotonic': time.monotonic()})


def stats_snapshot():
    return validate_stats(json.loads(subprocess.check_output(['sccache', '--show-stats', '--stats-format=json'], text=True)))


def validate_stats(payload):
    fields = ('compile_requests', 'requests_executed', 'compilations', 'cache_writes',
              'cache_read_errors', 'cache_write_errors', 'cache_hits', 'cache_misses', 'cache_errors',
              'non_cacheable_calls', 'non_cacheable_compilations', 'non_compilation_calls', 'compilation_failures', 'cache_timeouts', 'not_cached', 'not_cacheable')
    stats = {key: payload['stats'][key] for key in fields if key in payload['stats']}
    def numeric(value):
        if isinstance(value, dict):
            require(len(value) <= 128, 'oversized counter map')
            for key, child in value.items():
                require(isinstance(key, str) and len(key) <= 128, 'invalid counter name')
                numeric(child)
        else:
            require(type(value) is int and 0 <= value <= 2**53-1, 'invalid counter value')
    numeric(stats)
    for key in ('compile_requests','requests_executed','cache_read_errors','cache_write_errors','cache_hits','cache_misses','cache_errors','non_cacheable_calls','non_cacheable_compilations','non_compilation_calls'):
        require(key in stats, 'missing counter: '+key)
    return {'stats': stats}


def counts(raw, key):
    value = raw['stats'][key]['counts']
    require(isinstance(value, dict) and all(type(v) is int and v >= 0 for v in value.values()), 'malformed language counters')
    return value


def finish(directory):
    context = read(directory/'context.json')
    endpoint = time.monotonic()
    action_seconds = endpoint - read(directory/'start.json')['monotonic']
    raw = stats_snapshot()
    save(directory/'raw-stats.json', raw)
    hits, misses = counts(raw, 'cache_hits'), counts(raw, 'cache_misses')
    total = sum(hits.values()) + sum(misses.values())
    require(total > 0, 'no cacheable requests: inconclusive')
    require(os.environ.get('CANARY_BUILD_OUTCOME') == 'success', 'package verification failed')
    cmake = Path(BUILD_DIR)/'CMakeCache.txt'
    cmake_text = cmake.read_text()
    for compiler in ('C', 'CXX'):
        import re
        launcher = re.search(r'^CMAKE_'+compiler+r'_COMPILER_LAUNCHER:[^=]+=(.+)$', cmake_text, re.MULTILINE)
        require(launcher is not None and Path(launcher[1]).name=='sccache', 'native compiler launcher missing')
    save(directory/'native-build-evidence.json', {'cmake_cache_sha256': hashlib.sha256(cmake.read_bytes()).hexdigest(),
         'compiler_launchers': 'sccache', 'package_action_verified': True})
    require(raw['stats']['cache_read_errors'] == raw['stats']['cache_write_errors'] == 0, 'cache errors: inconclusive')
    require(sum(counts(raw, 'cache_errors').values()) == 0, 'cache errors: inconclusive')
    rate = sum(hits.values()) / total
    floor = context['arm'] != 'warm' or rate >= .01
    native_keys = ('C/C++', 'C', 'C++')
    native_hits = sum(hits.get(k, 0) for k in native_keys)
    native_requests = native_hits + sum(misses.get(k, 0) for k in native_keys)
    files = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('runtime-input').rglob('*') if p.is_file() and p.suffix in ('.json', '.sha256')}
    require(files, 'missing verified package manifests')
    save(directory/'result.json', {**context, 'action_seconds': action_seconds, 'native_preparation_and_build_seconds': None,
         'packaging_seconds': None, 'phase_timing_note': 'derive optional observed split from timestamped job log', 'classification': 'measured' if floor else 'warm-floor-failure',
         'warm_floor_passed': floor, 'hit_rate': rate, 'native_hits': native_hits, 'native_cacheable_requests': native_requests,
         'language_hits': hits, 'language_misses': misses,
         'assembler_hits': hits.get('Assembler', 0), 'assembler_misses': misses.get('Assembler', 0), 'manifest_and_checksum_hashes': files,
         'total_seconds': endpoint-read(directory/'restore-start.json')['monotonic'],
         'eligibility_changed': False, 'verified': True})
    if not floor:
        print('runtime seed canary: not qualified: warm 1% floor failed', file=sys.stderr)
        return 1


def summarize(directory):
    results = []
    for path in directory.rglob('result.json'):
        result = read(path)
        raw = validate_stats(read(path.parent/'raw-stats.json'))
        require(raw['stats']['cache_read_errors']==raw['stats']['cache_write_errors']==0, 'cache errors in sample')
        require(sum(counts(raw,'cache_errors').values())==0, 'cache errors in sample')
        hits,misses=counts(raw,'cache_hits'),counts(raw,'cache_misses')
        require(sum(hits.values())+sum(misses.values())>0, 'no cacheable requests')
        native_hits=sum(hits.get(k,0) for k in ('C/C++','C','C++'))
        native_requests=native_hits+sum(misses.get(k,0) for k in ('C/C++','C','C++'))
        require(result['native_hits']==native_hits and result['native_cacheable_requests']==native_requests, 'native counter mismatch')
        rate = sum(hits.values()) / (sum(hits.values()) + sum(misses.values()))
        floor = result['arm'] != 'warm' or rate >= .01
        derived = {'hit_rate': rate, 'warm_floor_passed': floor,
                   'classification': 'measured' if floor else 'warm-floor-failure',
                   'language_hits': hits, 'language_misses': misses,
                   'assembler_hits': hits.get('Assembler', 0), 'assembler_misses': misses.get('Assembler', 0)}
        for key, expected in derived.items():
            require(result.get(key) == expected, 'derived counter mismatch: '+key)
        results.append(result)
    require(len(results)==6, 'inconclusive: require all six results')
    indexed = {(x['pair'], x['arm']): x for x in results}
    require(set(indexed)=={(p,a) for p in (1,2,3) for a in ('cold','warm')}, 'duplicate/missing sample')
    for item in results:
        require(item.get('verified') is True and item.get('eligibility_changed') is False, 'missing verification')
        cache_identity({'actions_caches':[item['cache']]})
        require(item['image']==IMAGE and item['epoch']==EPOCH, 'image mismatch')
        for field in ('total_seconds','action_seconds','restore_seconds'):
            value=item.get(field)
            require(type(value) in (int,float) and math.isfinite(value) and value>=0, 'invalid timing')
        for field in ('native_hits','native_cacheable_requests'):
            require(type(item.get(field)) is int and item[field]>=0, 'invalid native counter')
        require(item['native_hits']<=item['native_cacheable_requests'], 'inconsistent native counters')
        require(type(item.get('warm_floor_passed')) is bool, 'missing warm floor result')
    identity = {(x['source'],x['image'],x['epoch'],x['cache']['id'],x['cache']['version']) for x in results}
    require(len(identity)==1 and all(x['classification'] in ('measured','warm-floor-failure') for x in results), 'inconclusive/mismatched evidence')
    same_run = {(x['run_id'],x['run_attempt']) for x in results}
    require(len(same_run)==1, 'samples from different attempts')
    for pair in (1,2,3):
        cold,warm=indexed[pair,'cold'],indexed[pair,'warm']
        require(cold['native_cacheable_requests'] > 0 and cold['native_cacheable_requests'] == warm['native_cacheable_requests'], 'incomparable C/C++ workloads')
        for key in ('host_cpu', 'runner_class', 'kernel'):
            require(cold.get(key) and cold[key] == warm.get(key), 'incomparable host: '+key)
        for key in ('runner_image_os', 'runner_image_version'):
            require(cold.get(key) == warm.get(key), 'incomparable host: '+key)
    deltas = [indexed[p,'cold']['total_seconds']-indexed[p,'warm']['total_seconds'] for p in (1,2,3)]
    output = {'schema':1,'paired_seconds_saved':deltas,'median_seconds_saved':statistics.median(deltas),
              'native_coverage_observed':all(indexed[p,'warm']['native_hits'] > indexed[p,'cold']['native_hits'] and
                  indexed[p,'warm']['native_cacheable_requests'] > 0 and
                  indexed[p,'warm']['native_cacheable_requests']-indexed[p,'warm']['native_hits'] <
                  indexed[p,'cold']['native_cacheable_requests']-indexed[p,'cold']['native_hits'] for p in (1,2,3)),
              'total_benefit_observed':all(d>0 for d in deltas),'eligibility_changed':False}
    reasons=[]
    if not all(x['warm_floor_passed'] for x in results): reasons.append('warm-floor-failure')
    if not output['native_coverage_observed']: reasons.append('no-incremental-c-cpp-coverage')
    if not output['total_benefit_observed']: reasons.append('no-consistent-total-time-benefit')
    output.update({'classification':'not-qualified' if reasons else 'observed-benefit', 'reasons':reasons})
    save(directory/'summary.json',output)
    print(json.dumps(output,indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['preflight','restore_start','restored','start','finish','summarize']);parser.add_argument('directory',type=Path);parser.add_argument('args',nargs=argparse.REMAINDER);args=parser.parse_args()
    try:
        return globals()[args.command](args.directory) or 0
    except (ValueError,KeyError,TypeError,OSError,subprocess.SubprocessError) as error:
        if args.command=='summarize':
            args.directory.mkdir(parents=True,exist_ok=True)
            save(args.directory/'summary.json', {'classification':'inconclusive','reason':str(error),'eligibility_changed':False})
        print('runtime seed canary: inconclusive: '+str(error),file=sys.stderr)
        return 1
if __name__=='__main__':sys.exit(main())
