# Large-Scale Real-World RL System Design - SOL v0

Date: 2026-05-18

## Goal

This document is a workload and speed-of-light (SOL) model for large-scale real-world robot training.

The purpose is not to specify an implementation. The purpose is to:

- identify the workload shape
- derive theoretical lower bounds
- expose first-order bottlenecks
- compare optimal data-movement patterns
- produce SOL tables for feasibility analysis

Primary target:

- 100 physical robots.
- 1 rollout GPU per robot.
- Rollout GPU and robot/env are co-located on the same rollout node.
- Separate train cluster and rollout cluster.
- Async RL workflow with pi-series policy, initially Pi0.5 3B BF16.
- Action chunk size is 50.
- Rollout uses real-time chunking (RTC).

Hard constraints:

- Robots must not stop.
- Policy/data staleness should be bounded, ideally 1 training-version off.
- Training throughput must keep up with robot data production.
- Hardware cost is not a v0 optimization target.

Non-goals for this version:

- exact software implementation
- RPC protocol selection
- process layout details
- production fault-handling design
- detailed transformer/FSDP layer-by-layer modeling

Implementation notes in this file should be read as constraints implied by the SOL model, not as a concrete build plan.

## Baseline Assumptions

| Parameter | Symbol | v0 value |
|---|---:|---:|
| Number of robots | `N` | 100 |
| Rollout mapping | - | 1 GPU : 1 robot |
| Rollout GPUs per physical node | `G` | default 8 |
| Rollout physical nodes | `M_rollout = ceil(N / G)` | 13 for `N=100,G=8` |
| Train GPUs | `T_gpu` | default 32, variable |
| Train GPUs per physical node | `G_train` | default 8 |
| Train physical nodes | `M_train = ceil(T_gpu / G_train)` | derived |
| Control frequency | `f` | numeric input, default 30Hz |
| Action chunk length | `C` | 50 steps |
| RTC stride | `S_rtc` | `C/2` for half-overlap RTC |
| Observation pipeline p99 | `T_obs_pipeline_p99` | default 0.010s |
| Chunk inference p99 | `T_infer_p99` | default 0.150s |
| Action pipeline p99 | `T_action_pipeline_p99` | default 0.010s |
| RTC jitter reserve p99 | `T_jitter_p99` | default 0.050s |
| Max trajectory horizon | `H` | 100 steps |
| Weight update cadence | `K` | update rollout weights every `K` horizons, default 1 |
| Replay granularity | `R_unit` | `step` or `chunk`; current RLinf pi0.5 is chunk-like |
| Step-level payload | `S_step` | 1MB per primitive step |
| Chunk-level payload | `S_chunk` | derived from boundary observations and chunk metadata |
| Trajectory payload | `S_episode` | depends on `R_unit` |
| Model sync payload | `W` | 8.5GB full state_dict reference |
| Model baseline | - | Pi0.5 3B BF16, no sharding |
| Weight sync policy | - | sync after each train update, initially full weights |
| Allowed policy version lag | `L` | 1, 2, 4, 8 versions |
| Cross-cluster bandwidth | `B_cross` | total aggregate effective bandwidth per direction; default symmetric full-duplex |

Notes:

- `W=8.5GB` comes from the observed RLinf Pi0.5 3B run. Pure BF16 parameters for 3B params would be about 6GB, but SOL should use the larger measured sync payload until proven otherwise.
- `S_step=1MB` is kept as the v0 step-level sizing point, but it should be derived from the observation payload model below rather than treated as a magic constant.
- Current RLinf pi0.5 real-world style replay is closer to `chunk` granularity: with `H=100` and `C=50`, one episode has about 2 replay entries, not 100.

## Not Yet Modeled

This section records workload factors that are intentionally outside the current SOL tables. They can change the effective throughput or flip a "green" SOL result to "red", so they need explicit follow-up before final system sizing.

### Robot / Environment Lifecycle

| Factor | Why it matters | Current treatment |
|---|---|---|
| Real-robot reset after success/failure | Reset consumes wall-clock time and robot capacity; effective sample rate is lower than `N * f` if reset is frequent or slow | Not modeled |
| Reset scheduling vs replay/weight traffic | Reset may need commands, observations, safety checks, and status messages that contend with control/replay/weight paths | Not modeled |
| Episode length distribution | Current tables assume max horizon `H=100`; early success/failure changes trajectory rate and reset frequency | Simplified as fixed `H` |
| Task setup variability | Object placement, scene reset, fixture motion, human cleanup, or automated reset stations can dominate robot utilization | Not modeled |
| System fault tolerance / downtime | Robot faults, GPU/node failures, network partitions, replay service failures, train job failures, E-stop, calibration, maintenance, and recovery reduce effective capacity | Not modeled; treated as an availability factor outside v0 SOL |

Reset should eventually enter the SOL model as:

```text
T_episode = H_effective / f
T_cycle = T_episode + T_reset
effective_robot_utilization = T_episode / T_cycle
effective_env_steps_per_sec = N * f * effective_robot_utilization
```

System fault tolerance should eventually enter as an availability multiplier:

```text
N_effective = N * availability
effective_env_steps_per_sec = N_effective * f * effective_robot_utilization
```

But v0 does not model HA design, failover, checkpoint recovery, replay durability, scheduler resubmission, or degraded-mode operation.

### Rollout Critical Path

| Factor | Why it matters | Current treatment |
|---|---|---|
| Pi0.5 chunk inference p50/p99 | RTC non-stop depends on chunk generation fitting inside RTC lookahead, not full chunk coverage | Modeled as a required measurement, not estimated |
| RTC overlap / stride | With half-overlap RTC, effective lookahead is `(C/2)/f`, not `C/f` | Modeled with `S_rtc=C/2` |
| 90Hz with small effective chunks | At 90Hz, `C=50` gives only 0.28s lookahead under half-overlap RTC | Partially modeled through RTC lookahead |
| 1 rollout rank : N robots | Fan-in observations and fan-out actions introduce batching barriers; GPU cost may drop but per-robot latency may not | Out of scope for v0; v0 assumes 1 GPU : 1 robot |
| Sensor pipeline jitter | Camera exposure, frame sync, dropped frames, USB/PCIe contention, compression CPU load, and timestamp skew affect RTC p99 | Modeled as `T_jitter_p99` input |
| Pack/unpack and ser/deser | H2D/D2H/PCIe/wire lower bounds are easy to derive, but packing, tensor layout, serialization format, and runtime boundaries are workload-specific | Kept inside measured `T_obs_pipeline` / `T_action_pipeline` buckets |
| Low-level controller limits | Servo loop behavior, actuator limits, safety filters, and robot firmware timing can cap real control frequency | Not modeled |

### Weight Activation / Policy Versioning

| Factor | Why it matters | Current treatment |
|---|---|---|
| `T_apply` / model activation time | Even if transfer is fast, loading, deserializing, validating, and swapping model weights can dominate version lag | Not modeled; must be measured |
| FSDP gather / state_dict materialization | Train cluster may need time to assemble publishable weights before network fanout starts | Parameterized through `T_train_update` / helper only |
| Inference engine swap behavior | SGLang/vLLM-style engines may have different reload/swap costs than direct PyTorch modules | Not modeled |
| Version activation safe point | RTC may only be able to activate a new model at chunk boundaries or other safe points | Not modeled beyond `T_rollout_activation` |

### Replay / Algorithmic Semantics

| Factor | Why it matters | Current treatment |
|---|---|---|
| Learning impact of chunk-level replay | Chunk replay is cheaper, but changes credit assignment and sample semantics | Performance-only model; algorithm quality not modeled |
| On-policy or near-on-policy constraints | If an algorithm requires `R_prod = R_cons` or low lag, train and rollout can become tightly synchronized | Only represented through `L` and `T_update` |
| Replay sampling/read path | Current tables focus on rollout-to-train ingest, not replay-buffer-to-trainer sampling/decode bandwidth | Not modeled |
| Reward model / evaluator overhead | Online reward inference, VLM scoring, or human labels can dominate per-step or per-episode cost | Not modeled |
| Eval traffic | Periodic evaluation rollouts and validation data are not included in the 100-robot training traffic | Not modeled |

### System / Data Movement Overheads

| Factor | Why it matters | Current treatment |
|---|---|---|
| Serialization and tensor/message count | For MB-scale payloads, per-call setup and tensor count can dominate raw bytes | Only represented as payload overhead |
| Packing vs streaming | Merging many tensors can reduce per-call overhead; streaming can overlap wire time | Represented as measured RTC pipeline overhead, not a pure bandwidth formula |
| Cross-cluster fabric jitter and packet loss | The SOL model assumes symmetric full-duplex per-direction bandwidth; endpoint contention and oversubscription can still create p99 stalls | Not modeled beyond `B_cross` |
| CPU contention on rollout nodes | Compression, replay export, model activation, camera ingest, and control tasks can compete | Not modeled |
| Cold logging / audit path | Step-level logs, video retention, and safety audit trails can be much larger than hot replay | Not modeled except storage caveat |
| Clock sync and timestamp accuracy | Versioning, replay freshness, and cross-node timelines require synchronized clocks | Not modeled |

Reference from `claude_notes/comm_path_breakdown_2node_1rank_NV_v3.pptx` final slides:

- real-robot reset path needs its own timeline capture
- 1 rollout rank : N robots introduces fan-in/fan-out and batching barriers
- 90Hz small-chunk mode may be inference-budget dominated
- pre/post-transport instrumentation is missing for model apply/reload and FSDP gather
- algorithm/stage performance curves can determine where producer/consumer rates stop scaling

## Step Payload Derivation

For step-level replay upload sizing, use:

```text
S_step = K_obs * V * H_img * W_img * 3 * bytes_per_channel / compression_ratio
       + S_state_action_reward
       + S_model_aux
       + S_serialization
```

Where:

- `V` = number of camera views used for policy/replay.
- `K_obs` = 1 if each step stores only `obs_t`; 2 if each transition stores both `obs_t` and `obs_{t+1}`.
- `S_state_action_reward` is small for typical robot state/action/reward/done fields, usually KB-level.
- `S_model_aux` can include logprob, value, policy version, task id, language id, timestamps, intervention flags, etc.
- `S_serialization` covers framing, array headers, compression container overhead, alignment, and retry metadata.
- `bytes_per_channel` defaults to 1 for uint8 RGB, and should be set to 2 for uint16/fp16 or 4 for fp32 image-like tensors.

RLinf/OpenPI-style real-world configs usually have 2 image inputs for Pi-series real-world policies, but dual-arm or richer logging can use 3 camera views. The 1MB estimate is plausible if replay stores uint8 images with current/next observations, or stores 3 camera views with moderate metadata.

Raw uint8 image payload examples:

| Camera views | Resolution | Store `obs_t` only | Store `obs_t` + `obs_{t+1}` | With 10-20% overhead |
|---:|---:|---:|---:|---:|
| 2 | 224x224 | 0.30MB | 0.60MB | 0.66-0.72MB |
| 3 | 224x224 | 0.45MB | 0.90MB | 0.99-1.08MB |
| 2 | 256x256 | 0.39MB | 0.79MB | 0.86-0.94MB |
| 3 | 256x256 | 0.59MB | 1.18MB | 1.30-1.42MB |

Interpretation:

- `S_step=1MB` is a reasonable conservative midpoint for 2-3 camera real-world replay with current/next observations and metadata.
- If the upload path sends JPEG-compressed images only, `S_step` may be lower.
- If it sends raw RGB at higher resolution, depth, masks, or multi-camera dual-arm logs, `S_step` can exceed 1MB quickly.
- The SOL helper defaults metadata/framing overhead to `10%` as a practical sizing buffer. Set it to `0%` for a strict lower-bound image-only calculation.
- The SOL calculator should expose `S_step` directly and also expose the image-derived formula as an optional helper.

## Replay Granularity: Step vs Chunk

Replay granularity is a first-order systems variable.

Define:

```text
R_unit = step or chunk
E_step_episode = H
E_chunk_episode = ceil(H / C)

S_step  ~= O_curr + O_next + A + R + M
S_chunk ~= O_curr + O_next + C * (A + R) + M_chunk
```

Where:

- `O_curr`, `O_next` are boundary observations.
- `A` is one primitive action payload.
- `R` is one primitive reward/done/truncation metadata payload.
- `M` and `M_chunk` are per-entry metadata/framing overheads.

For image-heavy robot replay, `O_curr + O_next` dominates `C * (A + R)`, so chunk replay can reduce replay bytes, insert count, and message count by roughly the chunk factor `C`.

For `H=100`, `C=50`:

| Replay unit | Entries/episode | Observation-bearing entries | Primitive steps represented |
|---|---:|---:|---:|
| Step-level | 100 | 100 | 100 |
| Chunk-level | 2 | 2 | 100 |

The right performance normalization must report both:

- replay entries/s
- replay MB/s
- represented primitive env steps/s
- represented primitive env steps/MB

Current RLinf pi0.5 real-world style path is effectively chunk-level:

- rollout predicts a 50-action chunk
- env executes the chunk
- replay stores a chunk transition with boundary observations, chunked actions, and per-substep rewards/dones
- intermediate observations are not replay-uploaded unless explicitly logged

Source note:

- `codex_notes/sys_design/replaybuffer_step_vs_chunk.md`

## Proposed System Topology

```text
                         Train Cluster
+----------------------------------------------------------------+
| Trainer GPUs                                                   |
|   - consume latest trajectories                                |
|   - run train updates                                           |
|   - publish policy version K+1                                  |
|                                                                |
| Replay Ingest Shards                                           |
|   - receive per-robot streams                                   |
|   - enforce FIFO/version freshness                              |
|   - persist or spill to storage                                 |
|                                                                |
| Weight Publisher Group                                         |
|   - read trained weights once                                   |
|   - fan out to rollout racks/caches                             |
|   - track per-robot ACK/version                                 |
+------------------------------+---------------------------------+
                               |
              Replay data plane|        Weight data plane
        rollout -> train       |        train -> rollout
                               |
                         Rollout Cluster
+----------------------------------------------------------------+
| Rack / pod 0                                                   |
|   Weight cache / local fanout                                  |
|   Rollout node 0: GPU infer + robot 0 + RTC controller         |
|   Rollout node 1: GPU infer + robot 1 + RTC controller         |
|   ...                                                          |
|                                                                |
| Rack / pod K                                                   |
|   Weight cache / local fanout                                  |
|   Rollout nodes ...                                            |
+----------------------------------------------------------------+
```

Each rollout node should contain:

- Real-time robot control loop.
- Observation capture and preprocessing.
- GPU model inference service.
- RTC action chunk queue.
- Local trajectory spool, preferably NVMe-backed.
- Async replay upload worker.
- Async weight receiver.
- Double-buffered model activation manager.

Critical rule: robot control and RTC action consumption must never wait on replay upload, weight download, model deserialization, or trainer progress.

## Cluster Scale Variables

Rollout cluster scale is directly determined by robot count under the v0 `1 GPU : 1 robot` assumption:

```text
rollout_gpu_count = N
M_rollout = ceil(N / G)
```

For `N=100`:

| Rollout GPUs / physical node `G` | Rollout GPUs | Physical rollout nodes `M_rollout` |
|---:|---:|---:|
| 1 | 100 | 100 |
| 4 | 100 | 25 |
| 8 | 100 | 13 |

Train cluster scale is a variable. It should not be derived from robot count directly. In the SOL model, train scale only matters through:

```text
T_train_update
B_cross_down
T_weight_publish_start_gap
T_train_export_or_materialization
```

Useful train-side variables:

| Variable | Meaning | Main SOL effect |
|---|---|---|
| `T_gpu` | number of train GPUs | lowers `T_train_flops`, may increase/shift FSDP communication |
| `M_train` | number of train physical nodes | determines train internal network shape and publisher placement |
| `G_train` | train GPUs per node | determines intra-node vs inter-node FSDP communication |
| `B_train_intra` | train node local GPU fabric | affects FSDP and state_dict materialization |
| `B_train_inter` | train inter-node network | affects FSDP and multi-node training |
| `B_cross` | total aggregate effective cross-cluster bandwidth per direction | caps replay uplink and weight downlink |

Interpretation:

- Rollout sizing is workload-fixed by `N` and `G`.
- Train sizing is a tunable knob to hit a target `T_train_update`.
- The main SOL tables should accept `T_train_update` as an input; an expandable helper can estimate it from `T_gpu`, `M_train`, FSDP communication, and compute.
- Weight publishers may be colocated with train nodes or separated into a publishing tier. For the main SOL model, expose only the aggregate effective downlink `B_cross`, not the number of publisher links.

## Four Data Paths

### Path 1: Observation, Robot to Inference

This is local to each rollout node in v0.

Expected payload is included in `S_step=1MB` for trainer upload sizing. For the real-time loop, the important quantity is not cross-cluster bandwidth, but local capture/preprocess latency.

Requirement:

```text
p99(obs_capture + preprocess + enqueue) << 1 / f
```

Step intervals:

| Frequency | Step interval |
|---:|---:|
| 10Hz | 100.0ms |
| 30Hz | 33.3ms |
| 90Hz | 11.1ms |

90Hz is a hard real-time engineering regime. The robot controller should consume already-prepared actions from a local ring buffer; it should not call the model inline.

### Path 2: Action Chunk, Inference to Robot

Action payload is tiny. For 7D float32 actions:

```text
50 actions * 7 dims * 4 bytes = 1.4KB per chunk
```

The issue is not bulk bandwidth. The issue is end-to-end p99 latency against the RTC deadline.

Chunk coverage and RTC lookahead:

```text
T_chunk_coverage = C / f
T_rtc_lookahead = (C / 2) / f       # half-overlap default
```

SOL condition for non-stop operation:

```text
T_obs_pipeline_p99 + T_infer_p99 + T_action_pipeline_p99 + T_jitter_p99 <= T_rtc_lookahead
```

Latency path is a serial sum, not a throughput bottleneck max:

```text
T_obs_pipeline_p99 =
  T_capture_p99
+ T_preprocess_p99
+ T_obs_transport_lb
+ T_obs_h2d_or_pcie_lb
+ T_obs_pack_unpack_p99
+ T_obs_ser_deser_p99
+ T_obs_queue_p99

T_action_pipeline_p99 =
  T_action_transport_lb
+ T_action_d2h_or_pcie_lb
+ T_action_pack_unpack_p99
+ T_action_ser_deser_p99
+ T_robot_enqueue_p99
```

The lower-bound data movement terms are straightforward:

```text
T_transport_lb = payload_bits / link_bandwidth + fixed_link_latency
T_h2d_lb       = tensor_bytes / H2D_effective_bandwidth
T_d2h_lb       = tensor_bytes / D2H_effective_bandwidth
T_pcie_lb      = bytes / PCIe_effective_bandwidth
```

But `pack/unpack` and `ser/deser` should not be over-modeled in v1. They depend on tensor count, contiguity, Python/C++ boundaries, wire format, compression, zero-copy/pinned memory, and runtime batching. Keep them as measured overhead buckets.

For the current system-design purpose, obs/action communication is unlikely to dominate because RTC overlaps chunks and action payload is small. It should remain visible as a latency bucket, but the first-order bottlenecks are still inference p99, weight sync, replay FIFO ingest, and later train consume throughput.

SOL RTC implications:

- Maintain an action chunk ring buffer owned by the real-time robot controller.
- Inference service asynchronously produces chunks from recent observations.
- Use low-watermark refill, not synchronous inference from the robot tick.
- Keep model hot-swap outside the robot control path.
- If buffer approaches empty, use a bounded fallback policy: hold-last-safe, slow-stop, or task-specific safe action. This is a safety fallback, not the steady-state path.

Queue depth tradeoff:

- Larger queue protects against inference/network jitter.
- Larger queue also increases action staleness because the robot executes actions planned from older observations.
- v0 recommendation: target enough headroom for the measured RTC next-chunk critical path plus jitter. At 90Hz, `C=50` gives only about 0.28s of half-overlap lookahead, so this becomes a separate operating regime.

### Path 3: Replay Buffer, Rollout to Trainer

Replay bandwidth depends on `R_unit`, but the most important FIFO producer metric is horizon production rate:

```text
RB_produce_horizons_per_sec = N * f / H
RB_produce_steps_per_sec    = N * f
```

For the default `N=100`, `f=10Hz`, `H=100`, the real-robot fleet produces:

```text
10 horizons/s
1,000 primitive steps/s
```

Replay ingest should report replay entries, represented primitive steps, and represented primitive steps per MB. Chunk-level replay can have low insert count while still representing the full robot execution stream.

Current-input table shape:

| Replay unit | Horizons/s | Entries/horizon | Entries/s | Payload/entry | Ingest bandwidth | Represented steps/s | Represented steps/MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Step-level | `N*f/H` | `H` | `N*f` | `S_step` | `N*f*S_step` | `N*f` | `1/S_step` |
| Chunk-level | `N*f/H` | `ceil(H/C)` | `N*f/H*ceil(H/C)` | `S_chunk` | `N*f/H*ceil(H/C)*S_chunk` | `N*f` | `H/(ceil(H/C)*S_chunk)` |

Default example with `N=100`, `f=10Hz`, `H=100`, `C=50`, `S_step=S_chunk=1MB`:

| Replay unit | Horizons/s | Entries/horizon | Entries/s | Ingest bandwidth | Line rate | Represented steps/MB |
|---|---:|---:|---:|---:|---:|---:|
| Step-level | 10 | 100 | 1,000 | 1GB/s | 8Gbps | 1 |
| Chunk-level | 10 | 2 | 20 | 20MB/s | 0.16Gbps | 50 |

This chunk-level table assumes replay stores only boundary observations and chunk metadata. If the system logs all intermediate images for replay or offline audit, use the step-level table or add a separate cold logging path.

Episode burst size for the same default:

```text
step-level S_episode = 100 steps * 1MB = 100MB per robot
chunk-level S_episode = ceil(100 / 50) * 1MB = 2MB per robot

step-level synchronized burst = 100 robots * 100MB = 10GB
chunk-level synchronized burst = 100 robots * 2MB = 200MB
```

If every robot flushes at the same time, the train ingress sees that burst every `H/f`. The v0 design should assume continuous streaming into bounded FIFO queues rather than synchronized whole-trajectory uploads.

Design implication:

- Stream per step or per small chunk into local spool and upload continuously.
- Randomize or shard upload scheduling by robot ID.
- Use per-robot bounded FIFO queues with version/timestamp metadata.
- Trainer should prefer latest data and should discard or downweight overly stale data.
- Always report both replay entries/s and represented primitive steps/s. Chunk-level replay can look low-volume while still representing all robot execution.

Storage/day and long-retention sizing are not part of the main SOL tables. They belong in an optional retention appendix or a separate cold logging model. The main replay table should focus on FIFO producer pressure:

- Hot replay tier for fresh training data.
- Cold/logging tier with sampling, compression, or retention policy if audit/offline training requires it.

### Cross-Cluster Bandwidth

The main SOL model assumes symmetric full-duplex bandwidth between train and rollout clusters. `B_cross` is total aggregate effective bandwidth per direction, not per-link bandwidth:

```text
B_cross_up   = B_cross_down = B_cross_total_aggregate_per_direction
```

Replay and weight sync should be checked by direction, not summed by default:

| Traffic | Direction | Shape | SOL check |
|---|---|---|---|
| Replay producer path | rollout -> train | continuous FIFO producer traffic | `B_cross >= B_replay_up` |
| Weight sync | train -> rollout | bursty deadline-bound traffic | `B_cross >= B_weight_down` |

Where:

```text
B_replay_up = active_replay_MBps * 8 / 1000
B_weight_down = fanout_bytes_per_version * 8 / (L * T_update)
```

This is intentionally a speed-of-light abstraction. Because the two flows are opposite directions on a full-duplex link, they should not be simply added in the main model.

Still-out-of-scope caveats:

- Endpoint resources can contend even when the wire is full-duplex: NIC DMA, PCIe, host memory bandwidth, CPU networking stack, NCCL, and local storage.
- Oversubscribed or non-blocking assumptions must be validated against the real leaf/spine topology.
- Weight sync is bursty while replay is continuous; burst shaping may still be required to protect p99 freshness.
- Control/metadata traffic is small enough to omit from the SOL tables, but production implementation should protect it with queueing/QoS.

### Path 4: Weight Sync, Trainer to Rollout

This is the main SOL risk if full weights are synced after every train update.

There are two useful SOL paths:

1. Direct per-GPU fanout: train publishers send one full weight copy to every rollout GPU.
2. Per-node scatter plus intra-node allgather: train publishers send one full model copy per physical inference node, sharded across its `G` rollout GPUs, then GPUs allgather locally so each DP inference GPU ends with full weights.

The second path is the SOL architecture if rollout servers have multiple GPUs. A simpler root-receive plus local broadcast path is useful as a baseline, but it creates a root GPU/NIC/PCIe hotspot and should not be treated as the optimal design.

#### Direct Per-GPU Fanout

For `W=8.5GB` and `N=100`:

```text
total fanout bytes per version = 850GB
```

Receiver-side lower bound for one rollout GPU:

| Link per receiver | Ideal receive time for 8.5GB |
|---:|---:|
| 100Gbps | 0.68s |
| 200Gbps | 0.34s |
| 400Gbps | 0.17s |

But source/publisher aggregate egress dominates. In the main SOL model, use aggregate effective downlink bandwidth directly:

```text
T_fanout >= (N * W) * 8 / B_cross_down
```

Where:

- `B_cross_down` is the total aggregate effective train-to-rollout bandwidth in Gbps.
- A topology helper can later derive `B_cross_down` from link count, NIC rate, and efficiency, but the main SOL tables should not expose those as first-layer inputs.

Raw ideal fanout time for `N*W=850GB`:

| Effective downlink | Ideal direct fanout time |
|---:|---:|
| 100Gbps | 68.0s |
| 200Gbps | 34.0s |
| 400Gbps | 17.0s |
| 800Gbps | 8.5s |
| 1.6Tbps | 4.25s |

Planning should not use raw line rate. At 60-70% effective throughput, add roughly 1.4-1.7x.

#### Per-Node Scatter plus Intra-Node Allgather

If a physical inference node has `G` rollout GPUs, the train publishers should send one disjoint `W/G` shard to each GPU in the node. The node then runs an intra-node allgather through NVLink/NVSwitch/PCIe/NCCL so every GPU has the full DP inference model.

This keeps cross-cluster bytes at one model copy per physical inference node, while avoiding a single root GPU receiving the whole model.

Definitions:

```text
M = ceil(N / G)                   # number of physical rollout nodes
cross_cluster_bytes = M * W
cross_cluster_bytes_per_gpu = W / G
intra_node_allgather_recv_per_gpu = (G - 1) / G * W
intra_node_allgather_send_recv_per_gpu = 2 * (G - 1) / G * W
intra_node_allgather_aggregate = (G - 1) * W per full node per direction
```

Timing lower bounds:

```text
T_sync_direct = (N * W) * 8 / B_cross_down

T_sync_node_cross = (M * W) * 8 / B_cross_down
T_sync_node_local = (2 * (G - 1) / G * W) / B_local_allgather_bidirectional_GBps

T_sync_node_no_pipeline = T_sync_node_cross + T_sync_node_local
T_sync_node_pipeline_lb = max(T_sync_node_cross, T_sync_node_local)
```

The per-node path is not slower on the cross-cluster network. It reduces cross-cluster bytes by roughly `G`, but adds local allgather work. In this model, `B_local_allgather_bidirectional_GBps` is the per-GPU bidirectional send+recv aggregate bandwidth in GB/s. The important comparison is therefore cross time, local send+recv/GPU, local time, no-pipeline total, and pipelined lower bound.

For `N=100`, `W=8.5GB`:

| GPUs per inference node `G` | Physical rollout nodes `M` | Cross-cluster bytes/version | Cross-cluster bytes/GPU | Intra-node aggregate/version per direction | Cross-cluster reduction vs direct |
|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 850.0GB | 8.50GB | 0GB | 1.0x |
| 4 | 25 | 212.5GB | 2.13GB | 637.5GB | 4.0x |
| 8 | 13 | 110.5GB | 1.06GB | 739.5GB | 7.7x |

Optional sensitivity view for one update per 100-step horizon:

| Frequency | Direct, G=1 | Node scatter, G=4 | Node scatter, G=8 |
|---:|---:|---:|---:|
| 10Hz | 680Gbps | 170Gbps | 88Gbps |
| 30Hz | 2.0Tbps | 510Gbps | 265Gbps |
| 90Hz | 6.1Tbps | 1.5Tbps | 796Gbps |

The main calculator should instead use the current input `f` and `K`, where `T_update=K*H/f`.

Intra-node allgather is still real work. A full 8-GPU node moves this much per direction:

```text
(G - 1) * W = 7 * 8.5GB = 59.5GB
```

But this should be pipeline-parallel, not root-serialized. Per GPU, the allgather receive volume is:

```text
(G - 1) / G * W = 7/8 * 8.5GB = 7.44GB
```

With the calculator's local bandwidth convention, the time model uses bidirectional send+recv traffic:

```text
2 * (G - 1) / G * W = 14.88GB per GPU for G=8,W=8.5GB
```

Two useful lower-bound models:

| Local sync model | Lower-bound expression | 8-GPU example |
|---|---:|---:|
| Naive root broadcast, serialized | `(G - 1) * W / B_single_path` | `59.5GB / B_single_path` |
| SOL scatter + NCCL allgather | `2*((G - 1) / G * W) / B_bidirectional_allgather_GBps` | `14.88GB / B_bidirectional_allgather_GBps` |

Illustrative 8-GPU allgather lower bounds:

| Effective bidirectional allgather bandwidth/GPU | Lower bound for 14.88GB/GPU |
|---:|---:|
| 25GB/s | 0.60s |
| 50GB/s | 0.30s |
| 64GB/s | 0.23s |
| 200GB/s | 0.074s |
| 400GB/s | 0.037s |

`64GB/s` is the PCIe 4.0 x16 theoretical bidirectional aggregate: about 32GB/s each direction.

Interpretation:

- Per-node scatter materially reduces train-rollout network fanout.
- It does not remove the need to copy weights into every inference GPU.
- The SOL path optimizes critical path and hotspot placement, not aggregate bytes: every GPU still needs the full model.
- For 10Hz/30Hz, 8-GPU nodes with NCCL allgather are attractive.
- For 90Hz, PCIe-only allgather may be acceptable only if the effective p99 is measured and model activation is fast; NVLink/NVSwitch gives much more margin.
- If rollout servers are single-GPU edge boxes, this optimization does not apply.

## Clock Analysis

There are three clocks that should not be confused:

```text
T_step = 1 / f
T_chunk_coverage = C / f
T_rtc_lookahead = (C / 2) / f     # half-overlap RTC default
T_horizon = H / f
T_update = K * H / f
```

For the default `f=10Hz`, `C=50`, `H=100`, `K=1`:

| Clock | Period | Meaning |
|---|---:|---|
| Control step | 0.10s | Robot consumes one primitive action |
| RTC chunk coverage | 5.00s | 50 actions of coverage |
| RTC lookahead | 2.50s | half-overlap chunk generation budget |
| 100-step horizon | 10.00s | one max-length trajectory |
| Weight update period | 10.00s | one rollout weight update every `K=1` horizon |

RTC non-stop condition:

```text
T_obs_pipeline_p99 + T_infer_p99 + T_action_pipeline_p99 + T_jitter_p99 <= T_rtc_lookahead
```

`T_apply` is not on the robot tick if rollout uses versioned/double-buffered policy activation. It still contributes to policy-version latency:

```text
T_train_update + T_export + T_weight_sync + T_apply <= L * T_update
```

Full-weight sync every control step is impossible:

```text
100 robots * 8.5GB * f
```

At 10Hz this would be 8.5TB/s of weight traffic. At 90Hz it would be 76.5TB/s.

Therefore "1-step-off" must be defined as one training-version off, not one robot-control-step off, unless the sync payload is reduced from full model weights to a very small delta.

For SOL, define:

```text
L = allowed rollout policy lag in training versions
K = number of horizons per rollout policy update
T_update = K * H / f
T_publish <= L * T_update
deadline = L * T_update
required_B_cross_direct = direct_weight_bytes * 8 / deadline
required_B_cross_node_no_pipeline = node_scatter_bytes * 8 / (deadline - T_local_allgather)
```

`L=1` is the ideal 1-version-off target. Higher `L` values relax policy freshness but may be operationally necessary for full-model sync.

For the node-scatter path, `required_B_cross_node_no_pipeline` is infeasible when `deadline <= T_local_allgather`.

For the default current-input case `N=100`, `f=30Hz`, `H=100`, `K=1`, `W=8.5GB`, `G=8`, and `B_local=64GB/s`:

| Path | L | Deadline | Local time | Required `B_cross` |
|---|---:|---:|---:|---:|
| Direct per-GPU | 1 | 3.33s | N/A | 2.04Tbps |
| Direct per-GPU | 2 | 6.67s | N/A | 1.02Tbps |
| Direct per-GPU | 4 | 13.33s | N/A | 510Gbps |
| Direct per-GPU | 8 | 26.67s | N/A | 255Gbps |
| Per-node scatter + allgather, `G=8` | 1 | 3.33s | 0.23s | 285Gbps |
| Per-node scatter + allgather, `G=8` | 2 | 6.67s | 0.23s | 137Gbps |
| Per-node scatter + allgather, `G=8` | 4 | 13.33s | 0.23s | 67Gbps |
| Per-node scatter + allgather, `G=8` | 8 | 26.67s | 0.23s | 33Gbps |

Frequency sweeps such as 10/30/90Hz should be presented as optional sensitivity views, not as the main table. The main calculator should always use the current input `f`.

`Min L for sync lower bound` should use the pipelined lower-bound path time:

```text
T_sync_lb_direct = T_sync_direct
T_sync_lb_node = max(T_sync_node_cross, T_local_allgather)
T_sync_lb_selected = T_sync_lb_direct or T_sync_lb_node for selected path
minL_sync_lb = max(1, ceil(T_sync_lb_selected / T_update))
```

Capacity status in the calculator should reserve margin:

```text
OK    if required <= 0.75 * capacity
Tight if required <= capacity
Over  if required > capacity
```

## Rollout Weight Update Deadline

This table should only define the rollout weight update deadline. It should not mix replay FIFO producer metrics or future train consume throughput.

```text
T_horizon = H / f
T_update = K * H / f
T_deadline = L * T_update
```

For the default `N=100`, `f=10Hz`, `H=100`, `K=1`:

| Metric | Value |
|---|---:|
| Update every K horizons | 1 |
| Horizon duration | 10.00s |
| Weight update period | 10.00s |
| Allowed policy lag | L=1 |
| Version deadline | 10.00s |

The full latency condition remains:

```text
T_train_update + T_export + T_weight_sync + T_apply <= L * T_update
```

In v1, `T_train_update`, `T_export`, and `T_apply` are `N/A`; only `T_weight_sync` has a numerical lower bound from the Weight Sync table.

The replay FIFO consume condition is separate and should not be mixed into this table:

```text
RB_consume_horizons_per_sec >= RB_produce_horizons_per_sec
RB_consume_steps_per_sec    >= RB_produce_steps_per_sec
```

This separation matters:

- FIFO consume capacity determines whether replay backlog grows.
- Weight update latency determines whether rollout stays within the selected policy-version lag.

## Train Cluster Internal Helper

The main SOL model should treat the train cluster as a bounded producer of policy versions. Its public interface to the rest of the system is:

```text
T_train_update                 # time to produce one new policy version
W                              # model payload to publish
T_weight_publish_start_gap     # delay before publishers can start sending
B_cross_down                   # aggregate effective train-to-rollout weight downlink
```

Train-internal FSDP details matter because they determine `T_train_update`, but they should stay in an expandable helper model. The rollout/robot non-stop design should not depend on transformer-layer-level training details.

Useful lower-bound structure:

```text
T_train_update >= max(
  T_train_flops,
  T_fsdp_comm,
  T_data_load,
  T_optimizer_and_export
)
```

Where:

- `T_train_flops` depends on model size, sequence/action horizon, batch size, update epochs, and train GPU compute.
- `T_fsdp_comm` depends on FSDP group size, inter-node bandwidth, intra-node bandwidth, and sharding strategy.
- `T_data_load` depends on replay ingest, hot buffer placement, storage/cache, and batch assembly.
- `T_optimizer_and_export` includes optimizer step, checkpoint/state_dict materialization, and making the new policy version visible to weight publishers.

v0 design choice:

- Keep `T_train_update` as a top-level input in the main tables.
- Add an optional train helper later to estimate `T_train_update`.
- Use the helper to answer whether `L=1` is plausible, not to drive the rollout control design directly.

## RTC Non-Stop SOL Constraints

The non-stop robot constraint means the workload must have a local real-time action supply. This section describes timing separation required by the SOL model, not a specific process implementation.

Logical workload roles:

```text
Robot real-time role
  - owns 10/30/90Hz tick
  - reads action from lock-free/ring buffer
  - never performs model inference
  - never waits for network or weight sync

Inference role
  - reads latest observation snapshot
  - generates 50-action chunk
  - appends chunk to RTC buffer
  - reports chunk version and obs timestamp

Weight activation role
  - downloads new model version in background
  - validates checksum
  - loads into inactive model buffer
  - exposes a new model version at a safe activation point

Replay export role
  - reads local trajectory spool
  - uploads continuously to train ingest
  - backpressures only local disk retention, never robot control
```

Hard requirements:

- No weight sync on the RTC next-chunk critical path.
- No model loading on the robot tick.
- No trajectory upload on the robot tick.
- Every action/step is tagged with policy version and observation timestamp.
- Rollout node can continue for a bounded time if train cluster is unavailable.

Memory implication:

- Double-buffering model weights requires roughly 2x inference weight memory.
- For Pi0.5 3B BF16, plan for 12-17GB just for active + staging weights, plus runtime overhead.
- This is acceptable on modern 48GB/80GB class rollout GPUs, but should be measured with the real inference stack.

## SOL Architecture Implications

1. Use the collective fanout lower bound, not single-source direct fanout.

   A single publisher directly pushing full weights to 100 rollout GPUs is not the SOL path. The lower-bound path is hierarchical/collective fanout aligned with physical topology.

   If rollout inference nodes have multiple GPUs, prefer per-node scatter plus intra-node NCCL allgather. This reduces train-rollout network bytes by roughly `G`, balances receive load across GPUs, and avoids a root-GPU hotspot. The cost is local GPU fabric pressure and model activation coordination.

2. Check replay and weight bandwidth by direction.

   Replay is continuous rollout-to-train traffic. Weight sync is bursty train-to-rollout traffic. Under the SOL full-duplex assumption, the main table should check uplink and downlink separately rather than adding them.

   Endpoint contention, oversubscription, and p99 queueing remain caveats, but they are not first-layer SOL variables.

3. Treat 90Hz as a different operating mode.

   At 90Hz, a 50-action chunk covers only 0.56s, and half-overlap RTC gives about 0.28s of lookahead. Full-weight sync per update becomes the dominant system constraint. Use adapter/delta sync or relaxed publish cadence unless there is very large effective weight downlink.

4. Stream replay continuously.

   Do not wait for 100-step trajectory completion before network transfer. Whole-traj flush creates avoidable burstiness and increases data staleness.

   Treat replay granularity as a variable. Current RLinf pi0.5 style is chunk-level; step-level replay is a worst-case mode that can increase replay bytes and insert rate by roughly `C`.

5. Bound staleness explicitly.

   Use policy version IDs everywhere:

   - action chunk has `policy_version`
   - each step has `policy_version`
   - trajectory has min/max policy version
   - trainer tracks age by wall-clock and version distance
   - stale data is downweighted or dropped

6. Make robot non-stop local.

   Robot liveness should depend only on the rollout node and the local action buffer. It should not depend on train cluster, central replay, weight publisher, or remote storage.

7. Keep train internals as an expandable helper.

   The main system should consume `T_train_update` as an input. FSDP communication and train cluster topology matter, but only through the rate at which train can produce new policy versions.

## First Bottleneck Read

For 100 robots, the likely bottlenecks are:

1. Weight fanout for full Pi0.5 sync, especially at 30Hz/90Hz.
2. RTC inference p99 at 90Hz, because half-overlap on a 50-action chunk gives only 0.28s of lookahead.
3. Intra-node allgather and model activation p99 if using 8-GPU inference nodes.
4. Endpoint/topology contention if replay and weight sync share NICs, PCIe, host memory, or oversubscribed fabric resources.
5. Replay ingest if all 1MB/step data is persisted at 90Hz.
6. Data staleness if trajectory upload is whole-episode rather than streaming.

Chunk-level replay network is manageable for 100 robots, but step-level replay at high control frequency becomes large. Weight fanout is the dominant cross-cluster risk unless per-node scatter, relaxed lag, or smaller sync payloads are used.

## Tooling TODO

Build an interactive HTML SOL calculator after the formulas settle.

Initial inputs:

- `N`: robot count.
- `G`: rollout GPUs per physical inference node.
- `M_rollout`: derived rollout node count, `ceil(N / G)`.
- `T_gpu`: train GPU count, default 32.
- `G_train`: train GPUs per physical node, default 8.
- `M_train`: derived train physical node count, `ceil(T_gpu / G_train)`.
- `f`: numeric control frequency, default 30Hz.
- `C`: action chunk length.
- `H`: trajectory horizon.
- `K`: update rollout weights every `K` horizons, default 1.
- `R_unit`: replay granularity, step or chunk.
- `S_step`: replay payload per step.
- `S_chunk`: replay payload per chunk entry.
- Optional camera-derived replay payload helper: views, square resolution, current/next obs, bytes per channel, compression ratio, overhead.
- `W`: model sync payload.
- `L`: allowed policy version lag.
- `B_cross`: total aggregate effective train-rollout bandwidth per direction, default symmetric full-duplex.
- `B_local`: intra-node per-GPU bidirectional send+recv aggregate allgather bandwidth in GB/s; default PCIe 4.0 x16 theoretical is 64GB/s.
- `T_obs_pipeline_p99`, `T_infer_p99`, `T_action_pipeline_p99`, `T_jitter_p99`: serial RTC next-chunk critical-path timing buckets; defaults are 0.010s, 0.150s, 0.010s, and 0.050s.
- `T_train_update`, `T_export`, `T_apply`: policy-version latency placeholders until train-side modeling is added.
- Parameter hover help: explain symbol, unit, default assumption, and formula usage without expanding the visible form.
- Optional train helper inputs: FSDP group size, train intra/inter-node bandwidth, estimated train FLOPs, and export time.

Proposed page hierarchy:

1. Top-level summary.

   - First row: Rollout cluster, Train cluster, Replay Buffer, FIFO.
   - Second row: Weight downlink, Weight sync lower bound, Min L for sync lower bound, RTC lookahead.
   - Weight downlink required bandwidth.
   - Min L for sync lower bound.
   - Red/yellow/green feasibility indicators.

2. Cross-Cluster Bandwidth.

   - Fixed hardware budget.
   - `B_cross` as total aggregate effective bandwidth per direction.
   - Replay Buffer (FIFO): rollout -> train.
   - Weight Sync: train -> rollout.

3. Rollout critical path.

   - RTC timing: step interval, chunk coverage, half-overlap lookahead.
   - `T_obs_pipeline + T_infer + T_action_pipeline + jitter` p99 serial RTC next-chunk critical-path budget.
   - Robot non-stop condition.
   - Action staleness from chunk queue depth.

4. Replay Buffer (FIFO).

   - Producer-side FIFO rate from rollout.
   - Step vs chunk, horizons/s, entries/s, represented steps/s.
   - Consume-side train model remains TBD.

5. Weight Sync.

   - Path Cost: direct per-GPU vs per-node scatter + allgather.
   - Version Lag Requirement: L=1/2/4/8 required `B_cross`; `B_local` is treated as fixed hardware and appears as local time.

6. Rollout Weight Update Deadline.

   - `T_update = K * H / f`.
   - `T_deadline = L * K * H / f`.
   - Latency condition: `T_train_update + T_export + T_weight_sync + T_apply <= L*T_update`.
   - `T_train_update`, `T_export`, and `T_apply` are `N/A` in v1.

7. Expandable internal helper models.

   - Train cluster update time helper.
   - Inference node local allgather helper.
   - Camera-derived `S_step` helper.
   - Replay step-vs-chunk helper.
   - Storage retention helper.

Layout will be designed separately once the system model stabilizes.

## Open Questions

1. What is the measured p50/p99 time to generate one 50-action Pi0.5 chunk on the target rollout GPU?
2. Does RTC generate a new chunk only at low-watermark, at fixed chunk boundaries, or in a receding-horizon style more frequently than one chunk per 50 steps?
3. What `K` horizon cadence should be evaluated beyond the default `K=1`: 2, 4, 8, or task-dependent?
4. Is adapter-only / LoRA / delta sync acceptable for v1, or must v0 assume full weights forever?
5. Is train-rollout networking same datacenter, cross-datacenter, or cloud-edge?
6. How many rollout GPUs are in one physical inference node, and what local fabric connects them: PCIe-only, NVLink, or NVSwitch?
7. What train cluster scale should be evaluated: `T_gpu`, `M_train`, train GPUs per node, and train intra/inter-node bandwidth?
8. How many rollout nodes sit behind one rack/pod-level switch? This determines the natural weight fanout tree.
9. What is the train cluster's target `T_train_update`, and how much of it is FSDP communication vs compute vs data load?
10. Are cross-cluster links close to symmetric full-duplex and non-blocking in practice, or do endpoint/topology bottlenecks require a margin model?
11. Is the target replay semantics step-level, chunk-level, or chunk-level hot replay plus step-level cold logging?
12. What is the reset-time distribution after success/failure, and how much does reset reduce effective robot utilization?
13. What is `T_apply` for the target inference stack: transfer complete to model version active?
14. How does producer/consumer scaling behave as `N_env` grows: where does rollout, replay ingest, sampling, or train update first stall?
15. What availability assumption should be used for SOL capacity planning, given that full fault tolerance and disaster recovery are out of v0 scope?
