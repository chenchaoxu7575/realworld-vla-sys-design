# Replay Buffer Perf: Step vs Chunk

Date: 2026-05-18

## Goal

Clarify how `step-level` versus `chunk-level` replay changes the performance profile of the data path around `replay buffer`, with emphasis on RLinf's current `pi0.5` real-world style pipeline.

This note is for performance analysis, not algorithm selection. The main question is:

- What changes on the wire?
- What changes in replay sample count?
- What changes in total bytes and message frequency?
- How should `horizon=100`, `action_chunk=50` be interpreted when sizing the replay path?

## Short Answer

For RLinf's current `pi0.5 + action_chunk=50` embodied SAC-style path, the replay path is effectively `chunk-level`, not `step-level`.

That means:

- `env -> rollout` sends one observation at each chunk boundary, not all 50 intermediate observations.
- `replay buffer` stores one chunk transition per policy decision, not 50 primitive-step transitions.
- With `horizon=100` and `chunk=50`, one episode contributes about `100 / 50 = 2` replay entries, not 100.
- Each replay entry is "fat": it carries chunked action and per-substep reward/done arrays.
- From a systems perspective, replay traffic is dominated by observation payloads, so chunking usually reduces replay bytes and message rate by close to the chunk factor `C`, unless intermediate observations are also logged.

The key perf consequence is:

```text
step-level: many small replay samples
chunk-level: few large replay samples
```

For image-heavy robot pipelines, "few large samples" is usually much cheaper end-to-end.

## Current RLinf Semantics

The current RLinf real-world chunked path behaves as follows:

1. `rollout` predicts a chunk of future actions.
2. `env` executes the whole chunk via `chunk_step()`.
3. After the chunk completes, `env -> rollout` sends only:
   - `obs`
   - `final_obs`
4. If `collect_transitions=True`, rollout/trajectory bookkeeping stores:
   - `curr_obs` at chunk start
   - `next_obs` at chunk end
   - one chunk action payload
   - chunk rewards/dones/truncations/terminations
5. `TrajectoryReplayBuffer` stores the resulting `Trajectory` as-is with shape `[T, B, ...]`, where `T` is chunk count, not primitive-step count.

In other words, RLinf currently treats a whole action chunk as one replay-time transition unit.

## Two Replay Designs

### A. Step-Level Replay

Canonical transition:

```text
(s_t, a_t, r_t, s_{t+1}, done_t)
```

If the episode horizon is `H`, replay gets about `H` entries per environment.

Properties:

- Decision frequency = environment step frequency
- Replay insert frequency = environment step frequency
- Replay sample count per episode = `H`
- Best for fine-grained credit assignment
- Highest message rate and largest observation duplication

### B. Chunk-Level Replay

Canonical transition:

```text
(s_t, A_{t:t+C-1}, R_{t:t+C-1}, s_{t+C}, done_{t:t+C-1})
```

Where:

- `A_{t:t+C-1}` is a chunk of `C` primitive actions
- `R_{t:t+C-1}` is a vector or aggregation over `C` rewards

If the episode horizon is `H`, replay gets about:

```text
T_chunk = ceil(H / C)
```

entries per environment.

Properties:

- Decision frequency reduced by `C`
- Replay insert frequency reduced by `C`
- Replay sample count per episode reduced by `C`
- Coarser credit assignment
- Much lower observation traffic when observations dominate payload size

## Cost Model

Define:

- `H`: primitive-step horizon per episode
- `C`: action chunk length
- `O`: serialized observation size for one boundary observation
- `A`: serialized primitive action size
- `R`: reward/done/truncation metadata size per primitive step
- `M`: fixed per-entry metadata overhead

Then:

### Step-Level Replay

Per entry:

```text
S_step ~= O_curr + O_next + A + R + M
       ~= 2O + A + R + M
```

Per episode:

```text
Bytes_step_episode ~= H * (2O + A + R + M)
Entries_step_episode = H
```

### Chunk-Level Replay

Per entry:

```text
S_chunk ~= O_curr + O_next + C*A + C*R + M_chunk
        ~= 2O + C(A + R) + M_chunk
```

Per episode:

```text
Bytes_chunk_episode ~= ceil(H / C) * (2O + C(A + R) + M_chunk)
Entries_chunk_episode = ceil(H / C)
```

### Dominant Regime for Robot VLA Training

For real-world robot VLA training:

- `O` is usually image-dominated and large
- `A`, `R`, and scalar metadata are usually tiny relative to `O`

So in practice:

```text
2O >> C(A + R)
```

which implies:

```text
Bytes_step_episode   ~= H * 2O
Bytes_chunk_episode  ~= ceil(H / C) * 2O
```

Therefore chunk replay often reduces total replay bytes by roughly `C`.

This is the main systems reason chunk-level replay is attractive even before discussing algorithmic trade-offs.

## Example: `H=100`, `C=50`

### Step-Level

Replay entries per episode:

```text
100
```

Traffic intuition:

- 100 inserts into replay
- 100 state transitions
- 100 copies of `curr_obs`
- 100 copies of `next_obs`

### Chunk-Level

Replay entries per episode:

```text
ceil(100 / 50) = 2
```

Traffic intuition:

- 2 inserts into replay
- 2 chunk transitions
- 2 `curr_obs`
- 2 `next_obs`
- each entry also carries:
  - 50-step action chunk
  - 50 rewards
  - 50 done/termination/truncation flags

### Why the Total Traffic Still Drops a Lot

Even though each chunk entry is fatter, it does not carry 50 intermediate observations.

So:

```text
step-level episode bytes    ~ 100 * image payload
chunk-level episode bytes   ~   2 * image payload + small chunk metadata
```

For image-heavy replay, the action/reward vectors are second-order terms.

## Link-by-Link Perf Impact

The replay path should be analyzed as separate links rather than as one monolithic "buffer cost".

### 1. `env -> rollout`

Current RLinf chunked behavior:

- one `obs` per chunk boundary
- not 50 intermediate observations

Perf effect of chunking:

- message count reduced by about `C`
- observation transfer bytes reduced by about `C`

This matters even before data hits replay.

### 2. `rollout/env -> replay buffer`

This is the most important link for replay analysis.

Step-level:

- `H` inserts per episode
- many small entries
- high queue pressure
- high serialization/deserialization call count

Chunk-level:

- `ceil(H/C)` inserts per episode
- few larger entries
- much lower insert frequency
- lower per-sample indexing pressure

If replay is remote, chunking also reduces network round trips and per-message framing overhead.

### 3. `replay buffer -> actor`

This link has two different notions of "amount of data":

- replay entries sampled
- primitive control steps represented

Under chunk replay:

- sampled entry count is small
- but each entry covers `C` primitive steps

This means perf analysis must distinguish:

```text
samples/s          vs          primitive-steps/s
```

Otherwise replay may look artificially cheap or artificially data-starved depending on which unit is used.

## Why Chunk Replay Can Mislead Perf Analysis

Chunk replay is efficient, but it changes the unit of accounting.

If you compare designs only by "number of replay samples", chunk replay looks tiny:

```text
H=100, C=50 -> 2 samples instead of 100
```

But if you compare by "primitive environment steps represented", those 2 samples still cover 100 executed robot steps.

For performance work, always report both:

1. `entries/s`
2. `primitive_steps/s represented by replay`

Recommended derived metrics:

- `replay_entries_per_sec`
- `replay_MB_per_sec`
- `represented_env_steps_per_sec`
- `represented_env_steps_per_MB`
- `represented_env_steps_per_insert`

These expose whether the system is saving bytes by reducing semantic granularity, or actually improving transport efficiency.

## Practical Interpretation for RLinf Pi0.5

For the current RLinf `pi0.5` real-world path:

- the policy is chunked
- the env interaction is chunked
- the replay path is chunked
- the replay count is therefore low by design

So for `horizon=100`, `chunk=50`, it is expected that replay traffic looks "small" if measured by:

- insert count
- index growth
- trajectory count

This does not mean the system is idle. It means each replay entry summarizes a large amount of robot execution.

From a perf perspective, the dominant savings come from not duplicating 98 intermediate image observations that would exist in a step-level design.

## Replay Buffer Memory and Indexing Implications

Step-level replay stresses:

- replay index size
- sample bookkeeping
- per-entry Python/object overhead
- queue wakeups
- lock contention
- serialization call count

Chunk-level replay shifts cost toward:

- larger contiguous tensors per entry
- heavier per-entry decode when sampled
- lower overall index growth
- lower insertion frequency

For Python-heavy replay implementations, chunk-level often wins by more than raw byte reduction alone because it also reduces control-plane overhead.

## Observed RLinf Reference Point

The cross-node RLinf measurement in `claude_notes/v1_vs_v3_comparison.md` reports a replay-buffer-side trajectory transfer of roughly:

- `2.67 MB`
- `22 tensors`
- about `22-28 ms` cross-node GLOO latency

That measurement is useful as a systems calibration point:

- replay-path payloads are already in the low-MB regime even before moving to step-level storage
- network latency is not only about bytes, but also about tensor/message count

If a design moved from chunk replay to step replay without blob packing, the increase in tensor count and message frequency could become as important as the increase in total bytes.

## What Would Change Under Step-Level Replay

If RLinf were modified to use step-level replay for `pi0.5`, the main systems changes would be:

1. `env` would need to surface intermediate observations for each substep inside `chunk_step()`.
2. trajectory building would need to emit `C` replay transitions per chunk.
3. replay insert frequency would increase by about `C`.
4. replay index growth and sampling frequency would increase by about `C`.
5. total observation traffic would increase drastically unless aggressive compression or deduplication were added.

So the replay path would stop being "few fat chunk entries" and become "many observation-heavy step entries".

## Perf Takeaways

- For current RLinf `pi0.5` real-world training, replay should be modeled as `chunk-level`.
- With `H=100`, `C=50`, one episode contributes about `2` replay entries, not `100`.
- This dramatically reduces replay bytes, insert rate, and message rate because observations dominate payload size.
- The right normalization is not just `samples`; it is also `primitive steps represented`.
- If future work changes replay to step-level, the replay path may become a first-order bottleneck even if actor and rollout compute stay unchanged.

## Recommended Perf Reporting Format

When comparing step vs chunk designs, report:

| Metric | Why it matters |
|---|---|
| `chunk_size C` | Defines semantic granularity |
| `episode horizon H` | Defines primitive control horizon |
| `replay entries / episode` | Directly drives insert pressure |
| `MB / replay entry` | Per-message transport cost |
| `MB / episode` | Total replay traffic |
| `entries / sec` | Control-plane load |
| `represented primitive steps / sec` | True robot throughput |
| `tensors / replay message` | Important for GLOO / RPC overhead |
| `cross-node vs local` | Determines whether network latency dominates |

This reporting format makes it possible to compare apples to apples when one system stores 100 step transitions and another stores 2 chunk transitions for the same robot execution.
