overview 
整体参考 RLinf 的真机 aysnc RL 的框架，但在目前的 SOL(speed of light) 阶段不要被框架限制了，我们应该在理论上推到整个系统的关键参数和性能数字
1. 硬件的拓扑
两个集群，train + rollout, 各自有多个 GPU nodes，互相之间通过网络连接(TCP/GDR 具体是什么暂时不重要)
rollout 节点上的每张卡都会带一个机器人(当前是一个，未来可能是多个，这个初版也可以不讨论)，简单的说 inference 和 对应的 env(robot) 在同一个节点
2. workflow
Async RL 的训练模型，基于 pi 系列模型，模型规模不大，不需要 sharding
trainer: 从 rollout/env 拿 traj，然后进行训练，每训练 n step 后，会把模型weight 同步给 inference
rollout: 每组 infer-robot 各自做交互，模型到达 max_horizon(提前success)，把 traj 传给 trainer
3. 4 条通信链路
1. obs: robot -> infer
2. action_chunk: infer -> robot
3. Replayer buffer: FIFO 的通道 infer/robot -> trainer，需要管理 数据 staleness，以及网络带宽
4. weight sync: trainer -> infer, 下发速度影响到model 的 staleness


一些实测基于 RLinf，作为参考，在 sol design 完成后，比较实测和 sol，以及合理的性能外推
claude_notes/comm_path_breakdown_2node_1rank_NV_v3_build.js
claude_notes/v1_vs_v3_comparison.md