import type { ConfigurationDefaultsSetting } from '@/features/app-tabs/types'
import {
  HARDWARE_TOML_SECTION,
  MODEL_FIT_TOML_SECTION,
  THROUGHPUT_TOML_SECTION
} from './configuration-defaults-constants'

export const CONFIGURATION_DEFAULT_RUNTIME_SETTINGS = [
  {
    id: 'threads',
    categoryId: 'runtime',
    icon: 'cpu',
    label: 'CPU threads',
    description:
      'Sets the default CPU thread count. Use 0 for auto; 256 is a safe UI ceiling for general-purpose systems.',
    inheritedLabel: 'Inherited by placements without a thread override',
    tomlSection: THROUGHPUT_TOML_SECTION,
    mutability: 'restart-required',
    control: { kind: 'range', name: 'threads', value: '0', min: 0, max: 256, step: 1, unit: 'threads' }
  },
  {
    id: 'threads-batch',
    categoryId: 'runtime',
    icon: 'cpu',
    label: 'Batch threads',
    description:
      'Sets the thread count used for batching. Use 0 for auto; 256 is a safe UI ceiling for general-purpose systems.',
    inheritedLabel: 'Inherited by placements without a batch-thread override',
    tomlSection: THROUGHPUT_TOML_SECTION,
    mutability: 'restart-required',
    control: { kind: 'range', name: 'threads_batch', value: '0', min: 0, max: 256, step: 1, unit: 'threads' }
  },
  {
    id: 'continuous-batching',
    categoryId: 'runtime',
    icon: 'layers',
    label: 'Continuous batching',
    description: 'Choose whether the runtime should keep batching continuously when supported.',
    inheritedLabel: 'Inherited by placements without a batching override',
    tomlSection: THROUGHPUT_TOML_SECTION,
    mutability: 'restart-required',
    control: {
      kind: 'choice',
      name: 'continuous_batching',
      value: 'auto',
      presentation: 'segmented',
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'on', label: 'on' },
        { value: 'off', label: 'off' }
      ]
    }
  },
  {
    id: 'numa',
    categoryId: 'runtime',
    icon: 'cpu',
    label: 'NUMA policy',
    description: 'Choose the NUMA policy used when launching the runtime.',
    inheritedLabel: 'Inherited by placements without a NUMA override',
    visibility: 'advanced',
    tomlSection: THROUGHPUT_TOML_SECTION,
    mutability: 'restart-required',
    control: {
      kind: 'choice',
      name: 'numa',
      value: 'auto',
      presentation: 'select',
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'disabled', label: 'disabled' },
        { value: 'distribute', label: 'distribute' },
        { value: 'isolate', label: 'isolate' },
        { value: 'numactl', label: 'numactl' }
      ]
    }
  },
  {
    id: 'cpu-affinity',
    categoryId: 'runtime',
    icon: 'cpu',
    label: 'CPU affinity',
    description: 'Pin runtime threads to a specific CPU mask such as 0-3,8-11.',
    inheritedLabel: 'Inherited by placements without an affinity override',
    visibility: 'advanced',
    tomlSection: THROUGHPUT_TOML_SECTION,
    mutability: 'restart-required',
    control: { kind: 'text', name: 'cpu_affinity', value: '', placeholder: 'e.g. 0-3,8-11' }
  },
  {
    id: 'priority',
    categoryId: 'runtime',
    icon: 'gauge',
    label: 'Process priority',
    description: 'Set the scheduler priority or nice value for the runtime process.',
    inheritedLabel: 'Inherited by placements without a priority override',
    visibility: 'advanced',
    tomlSection: THROUGHPUT_TOML_SECTION,
    mutability: 'restart-required',
    control: { kind: 'text', name: 'priority', value: '', placeholder: 'e.g. 0 or normal' }
  },
  {
    id: 'poll',
    categoryId: 'runtime',
    icon: 'zap',
    label: 'Poll mode',
    description: 'Choose how the runtime polls for work when busy-waiting is available.',
    inheritedLabel: 'Inherited by placements without a poll override',
    visibility: 'advanced',
    tomlSection: THROUGHPUT_TOML_SECTION,
    mutability: 'restart-required',
    control: {
      kind: 'choice',
      name: 'poll',
      value: 'auto',
      presentation: 'segmented',
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'busy', label: 'busy' },
        { value: 'sleep', label: 'sleep' }
      ]
    }
  },
  {
    id: 'slot-prompt-similarity',
    categoryId: 'runtime',
    icon: 'gauge',
    label: 'Slot prompt similarity',
    description: 'Tune the similarity threshold used when comparing slot prompts before reuse.',
    inheritedLabel: 'Inherited by placements without a slot similarity override',
    visibility: 'advanced',
    tomlSection: THROUGHPUT_TOML_SECTION,
    mutability: 'restart-required',
    control: {
      kind: 'range',
      name: 'slot_prompt_similarity',
      value: '0.50',
      min: 0,
      max: 1,
      step: 0.01
    }
  },
  {
    id: 'gpu-layers',
    categoryId: 'runtime',
    icon: 'layers',
    label: 'GPU layers',
    description: 'Set the GPU layer count, or use auto. The backend also accepts -1 to mean all layers.',
    inheritedLabel: 'Inherited by placements without a GPU layer override',
    tomlSection: HARDWARE_TOML_SECTION,
    mutability: 'restart-required',
    control: {
      kind: 'text',
      name: 'gpu_layers',
      value: 'auto',
      placeholder: 'auto or integer layer count'
    }
  },
  {
    id: 'mmap',
    categoryId: 'runtime',
    icon: 'memory',
    label: 'Memory map',
    description: 'Choose whether model files are memory-mapped when loaded.',
    inheritedLabel: 'Inherited by placements without a memory-map override',
    visibility: 'advanced',
    tomlSection: HARDWARE_TOML_SECTION,
    mutability: 'restart-required',
    control: {
      kind: 'choice',
      name: 'mmap',
      value: 'auto',
      presentation: 'segmented',
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'on', label: 'on' },
        { value: 'off', label: 'off' }
      ]
    }
  },
  {
    id: 'mlock',
    categoryId: 'runtime',
    icon: 'shield',
    label: 'Memory lock',
    description: 'Choose whether loaded model pages should be locked into RAM.',
    inheritedLabel: 'Inherited by placements without a memory-lock override',
    visibility: 'advanced',
    tomlSection: HARDWARE_TOML_SECTION,
    mutability: 'restart-required',
    control: {
      kind: 'choice',
      name: 'mlock',
      value: 'off',
      presentation: 'toggle',
      options: [
        { value: 'on', label: 'on' },
        { value: 'off', label: 'off' }
      ]
    }
  },
  {
    id: 'warmup',
    categoryId: 'runtime',
    icon: 'zap',
    label: 'Warmup',
    description: 'Choose whether the runtime should perform a warmup pass after load.',
    inheritedLabel: 'Inherited by placements without a warmup override',
    visibility: 'advanced',
    tomlSection: HARDWARE_TOML_SECTION,
    mutability: 'restart-required',
    control: {
      kind: 'choice',
      name: 'warmup',
      value: 'auto',
      presentation: 'segmented',
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'on', label: 'on' },
        { value: 'off', label: 'off' }
      ]
    }
  },
  {
    id: 'direct-io',
    categoryId: 'runtime',
    icon: 'folder',
    label: 'Direct I/O',
    description: 'Choose whether model files are opened with direct I/O when supported.',
    inheritedLabel: 'Inherited by placements without a direct I/O override',
    visibility: 'advanced',
    tomlSection: HARDWARE_TOML_SECTION,
    mutability: 'restart-required',
    control: {
      kind: 'choice',
      name: 'direct_io',
      value: 'off',
      presentation: 'toggle',
      options: [
        { value: 'on', label: 'on' },
        { value: 'off', label: 'off' }
      ]
    }
  },
  {
    id: 'split-mode',
    categoryId: 'runtime',
    icon: 'layers',
    label: 'Split mode',
    description: 'Choose how layers are split across devices when model sharding is enabled.',
    inheritedLabel: 'Inherited by placements without a split-mode override',
    visibility: 'advanced',
    tomlSection: HARDWARE_TOML_SECTION,
    mutability: 'restart-required',
    control: {
      kind: 'choice',
      name: 'split_mode',
      value: 'auto',
      presentation: 'select',
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'none', label: 'none' },
        { value: 'layer', label: 'layer' },
        { value: 'row', label: 'row' }
      ]
    }
  },
  {
    id: 'main-gpu',
    categoryId: 'runtime',
    icon: 'server',
    label: 'Main GPU',
    description: 'Select the primary GPU index used for loading and dispatch.',
    inheritedLabel: 'Inherited by placements without a main-GPU override',
    visibility: 'advanced',
    tomlSection: HARDWARE_TOML_SECTION,
    mutability: 'restart-required',
    control: { kind: 'range', name: 'main_gpu', value: '0', min: 0, max: 7, step: 1, unit: 'GPU index' }
  },
  {
    id: 'tensor-split',
    categoryId: 'runtime',
    icon: 'layers',
    label: 'Tensor split',
    description: 'Set the tensor split ratios for multi-GPU placement, for example 0.5,0.5.',
    inheritedLabel: 'Inherited by placements without a tensor-split override',
    visibility: 'advanced',
    tomlSection: HARDWARE_TOML_SECTION,
    mutability: 'restart-required',
    control: { kind: 'text', name: 'tensor_split', value: '', placeholder: 'e.g. 0.5,0.5' }
  },
  {
    id: 'parallel-slots',
    categoryId: 'runtime',
    tomlSection: 'defaults.throughput',
    tomlKey: 'parallel',
    icon: 'cpu',
    label: 'Default slots / parallel requests',
    description:
      'Sets the default parallel slots for placements without their own value. More slots increase KV memory use.',
    inheritedLabel: 'Inherited by placements without a parallel override',
    control: { kind: 'range', name: 'parallel', value: '4', min: 1, max: 16, step: 1, unit: 'slots' }
  },
  {
    id: 'tuning-profile',
    categoryId: 'runtime',
    tomlSection: 'defaults.throughput',
    tomlKey: 'tuning_profile',
    icon: 'gauge',
    label: 'Default tuning profile',
    description: 'Choose the starting balance between throughput, batch size, and memory use.',
    inheritedLabel: 'Reset placements to default when experiments are finished',
    control: {
      kind: 'choice',
      name: 'tuning_profile',
      value: 'balanced',
      options: [
        { value: 'balanced', label: 'balanced' },
        { value: 'throughput', label: 'throughput' },
        { value: 'saver', label: 'saver' }
      ]
    }
  },
  {
    id: 'flash-attention',
    categoryId: 'runtime',
    tomlSection: 'defaults.model_fit',
    tomlKey: 'flash_attention',
    icon: 'layers',
    label: 'Flash attention policy',
    description: 'Choose the default attention kernel policy for compatible runtimes.',
    inheritedLabel: 'Inherited from Defaults unless a deployment pins kernels',
    control: {
      kind: 'choice',
      name: 'flash_attention',
      value: 'auto',
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'enabled', label: 'enabled' },
        { value: 'disabled', label: 'disabled' }
      ]
    }
  },
  {
    id: 'hardware-device',
    categoryId: 'runtime',
    tomlSection: 'defaults.hardware',
    tomlKey: 'device',
    icon: 'cpu',
    label: 'Default GPU device',
    description: 'Optional fallback device for pinned GPU assignment when a model does not set its own device.',
    inheritedLabel: 'Used only by placements without a model-specific hardware.device',
    control: { kind: 'text', name: 'device', value: '', placeholder: 'cuda:0 or CUDA0' }
  },
  {
    id: 'kv-cache',
    categoryId: 'memory',
    tomlSection: 'defaults.model_fit',
    tomlKey: 'kv_cache_policy',
    icon: 'filter',
    label: 'KV cache policy',
    description: 'Select how aggressively KV cache precision is reduced to fit larger contexts.',
    inheritedLabel: 'Used when the placement has no cache override',
    control: {
      kind: 'choice',
      name: 'kv_cache_policy',
      value: 'auto',
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'quality', label: 'quality' },
        { value: 'balanced', label: 'balanced' },
        { value: 'saver', label: 'saver' }
      ]
    }
  },
  {
    id: 'memory-margin',
    categoryId: 'memory',
    tomlSection: 'defaults.hardware',
    tomlKey: 'safety_margin_gb',
    icon: 'memory',
    label: 'Memory / safety margin',
    description: 'Keep this much GPU memory free before placement fit checks pass.',
    inheritedLabel: 'Applied before per-model fit checks',
    control: { kind: 'range', name: 'safety_margin_gb', value: '2', min: 0, max: 8, step: 0.5, unit: 'GB' }
  },
  {
    id: 'ctx-size',
    categoryId: 'memory',
    icon: 'gauge',
    label: 'Context window size',
    description: 'Set the default context window size in tokens.',
    inheritedLabel: 'Applied when a placement does not override context size',
    tomlSection: MODEL_FIT_TOML_SECTION,
    mutability: 'restart-required',
    control: { kind: 'range', name: 'ctx_size', value: '2048', min: 2048, max: 262144, step: 512, unit: 'tokens' }
  },
  {
    id: 'batch',
    categoryId: 'memory',
    icon: 'layers',
    label: 'Batch size',
    description: 'Set the default prefill batch size.',
    inheritedLabel: 'Applied when a placement does not override batch size',
    tomlSection: MODEL_FIT_TOML_SECTION,
    mutability: 'restart-required',
    control: { kind: 'range', name: 'batch', value: '512', min: 32, max: 4096, step: 32, unit: 'tokens' }
  },
  {
    id: 'ubatch',
    categoryId: 'memory',
    icon: 'layers',
    label: 'Micro-batch size',
    description:
      'Set the default micro-batch (physical prefill chunk) size. Values at or below 128 keep the CUDA SSM sequential-scan fallback; larger values enable the SSD chunked kernel for recurrent models.',
    inheritedLabel: 'Applied when a placement does not override micro-batch size',
    visibility: 'advanced',
    tomlSection: MODEL_FIT_TOML_SECTION,
    mutability: 'restart-required',
    control: { kind: 'range', name: 'ubatch', value: '512', min: 32, max: 4096, step: 32, unit: 'tokens' }
  },
  {
    id: 'cache-type-k',
    categoryId: 'memory',
    icon: 'filter',
    label: 'KV cache type (K)',
    description: 'Choose the KV cache dtype used for keys.',
    inheritedLabel: 'Applied when a placement does not override key cache dtype',
    tomlSection: MODEL_FIT_TOML_SECTION,
    mutability: 'restart-required',
    control: {
      kind: 'choice',
      name: 'cache_type_k',
      value: 'f16',
      presentation: 'segmented',
      options: [
        { value: 'f16', label: 'f16' },
        { value: 'q8_0', label: 'q8_0' },
        { value: 'q4_0', label: 'q4_0' }
      ]
    }
  },
  {
    id: 'cache-type-v',
    categoryId: 'memory',
    icon: 'filter',
    label: 'KV cache type (V)',
    description: 'Choose the KV cache dtype used for values.',
    inheritedLabel: 'Applied when a placement does not override value cache dtype',
    tomlSection: MODEL_FIT_TOML_SECTION,
    mutability: 'restart-required',
    control: {
      kind: 'choice',
      name: 'cache_type_v',
      value: 'f16',
      presentation: 'segmented',
      options: [
        { value: 'f16', label: 'f16' },
        { value: 'q8_0', label: 'q8_0' },
        { value: 'q4_0', label: 'q4_0' }
      ]
    }
  },
  {
    id: 'kv-offload',
    categoryId: 'memory',
    icon: 'server',
    label: 'KV offload',
    description: 'Choose whether KV cache offloading stays enabled.',
    inheritedLabel: 'Applied when a placement does not override KV offload',
    visibility: 'advanced',
    tomlSection: MODEL_FIT_TOML_SECTION,
    mutability: 'restart-required',
    control: {
      kind: 'choice',
      name: 'kv_offload',
      value: 'auto',
      presentation: 'segmented',
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'on', label: 'on' },
        { value: 'off', label: 'off' }
      ]
    }
  },
  {
    id: 'kv-unified',
    categoryId: 'memory',
    icon: 'memory',
    label: 'Unified KV',
    description: 'Choose whether sequences share one unified KV buffer.',
    inheritedLabel: 'Applied when a placement does not override unified KV',
    visibility: 'advanced',
    tomlSection: MODEL_FIT_TOML_SECTION,
    mutability: 'restart-required',
    control: {
      kind: 'choice',
      name: 'kv_unified',
      value: 'auto',
      presentation: 'segmented',
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'on', label: 'on' },
        { value: 'off', label: 'off' }
      ]
    }
  },
  {
    id: 'cache-ram-mib',
    categoryId: 'memory',
    icon: 'memory',
    label: 'Prompt cache RAM',
    description: 'Set the maximum prompt cache size in MiB. Use 0 to disable the cache.',
    inheritedLabel: 'Applied when a placement does not override prompt cache RAM',
    visibility: 'advanced',
    tomlSection: MODEL_FIT_TOML_SECTION,
    mutability: 'restart-required',
    control: { kind: 'range', name: 'cache_ram_mib', value: '8192', min: 0, max: 65536, step: 256, unit: 'MiB' }
  },
  {
    id: 'cache-idle-slots',
    categoryId: 'memory',
    icon: 'layers',
    label: 'Idle slot caching',
    description: 'Save and clear idle slots when a new task starts; requires unified KV and cache RAM.',
    inheritedLabel: 'Applied when a placement does not override idle slot caching',
    visibility: 'advanced',
    tomlSection: MODEL_FIT_TOML_SECTION,
    mutability: 'restart-required',
    control: { kind: 'range', name: 'cache_idle_slots', value: '4', min: 0, max: 64, step: 1, unit: 'slots' }
  },
  {
    id: 'prompt-cache',
    categoryId: 'memory',
    icon: 'filter',
    label: 'Prompt cache',
    description: 'Choose whether prompt caching stays auto-managed or explicitly on or off.',
    inheritedLabel: 'Applied when a placement does not override prompt caching',
    visibility: 'advanced',
    tomlSection: MODEL_FIT_TOML_SECTION,
    mutability: 'restart-required',
    control: {
      kind: 'choice',
      name: 'prompt_cache',
      value: 'auto',
      presentation: 'segmented',
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'on', label: 'on' },
        { value: 'off', label: 'off' }
      ]
    }
  },
  {
    id: 'context-shift',
    categoryId: 'memory',
    icon: 'layers',
    label: 'Context shift',
    description: 'Allow context shifting for long-running generations when supported, or leave it on auto.',
    inheritedLabel: 'Applied when a placement does not override context shift',
    visibility: 'advanced',
    tomlSection: MODEL_FIT_TOML_SECTION,
    mutability: 'restart-required',
    control: {
      kind: 'choice',
      name: 'context_shift',
      value: 'auto',
      presentation: 'segmented',
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'on', label: 'on' },
        { value: 'off', label: 'off' }
      ]
    }
  }
] as const satisfies readonly ConfigurationDefaultsSetting[]
