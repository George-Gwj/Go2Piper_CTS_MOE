# AMP 接入说明

本文档记录当前在 CTS-MoE PPO 训练中新增的 AMP（Adversarial Motion Priors）实现方案和代码接入点。

## 目标

AMP 用 `datasets/mocap_motions/*` 中的 mocap trot 数据训练一个 discriminator，给 policy rollout 额外提供运动风格奖励。

当前实现只约束腿部运动风格，不把手臂状态放入 AMP discriminator。

最终 PPO reward 为：

```python
total_reward = task_reward + amp_reward
```

`amp_task_reward_lerp` 已作为配置项保留，但当前没有参与 reward 混合。

## AMP Observation

每帧 AMP observation 为 30 维：

```text
joint_pos(12) + base_lin_vel(3) + base_ang_vel(3) + joint_vel(12)
```

mocap 文件每帧为 61 维，当前切片规则为：

```python
joint_pos = frame[7:19]
base_lin_vel = frame[31:34]
base_ang_vel = frame[34:37]
joint_vel = frame[37:49]
amp_obs = cat([joint_pos, base_lin_vel, base_ang_vel, joint_vel])
```

仿真侧从 `robot.data` 中按同样顺序提取：

```python
robot.data.joint_pos[:, amp_joint_ids]
robot.data.root_lin_vel_b
robot.data.root_ang_vel_b
robot.data.joint_vel[:, amp_joint_ids]
```

腿关节顺序为：

```text
FR_hip_joint, FR_thigh_joint, FR_calf_joint,
FL_hip_joint, FL_thigh_joint, FL_calf_joint,
RR_hip_joint, RR_thigh_joint, RR_calf_joint,
RL_hip_joint, RL_thigh_joint, RL_calf_joint
```

注意：当前仿真侧使用绝对 `joint_pos`，没有减 `default_joint_pos`。如果后续 AMP reward 长期异常偏低，需要优先检查 mocap 和仿真关节角参考系是否一致。

## Discriminator

新增文件：

```text
scripts_attnnopriv/rsl_rl/local_rsl_rl/modules/amp_discriminator.py
```

网络结构：

```text
input 60
Linear 60 -> 1024 -> ReLU
Linear 1024 -> 512 -> ReLU
Linear 512 -> 1
```

输入为连续两帧拼接：

```python
disc_input = cat([amp_obs_t, amp_obs_t1])  # 30 + 30 = 60
```

训练目标：

```text
expert_d -> +1
policy_d -> -1
```

损失：

```python
expert_loss = MSE(expert_d, +1)
policy_loss = MSE(policy_d, -1)
amp_loss = 0.5 * (expert_loss + policy_loss)
amp_loss += amp_grad_penalty_lambda * grad_penalty
```

当前 gradient penalty 在 expert transition 上计算 discriminator 输出对输入的梯度平方和。

## AMP Reward

policy transition 的 AMP reward 为：

```python
amp_reward = amp_reward_coef * clamp(
    1.0 - 0.25 * (D(policy_transition) - 1.0) ** 2,
    min=0.0,
)
```

当前默认：

```python
amp_reward_coef = 0.01
```

因此 AMP reward 最大值为 `0.01`。

## Mocap Dataset

新增文件：

```text
scripts_attnnopriv/rsl_rl/local_rsl_rl/modules/amp_motion_dataset.py
```

功能：

- 支持 `datasets/mocap_motions/*` glob。
- 读取 mocap JSON 单行文件。
- 将每帧转换为 30 维 AMP obs。
- 支持 wrap-around transition 采样。
- 支持预采样 replay buffer。

当前参数：

```text
amp_replay_buffer_size      = 1,000,000
amp_num_preload_transitions = 2,000,000
```

实际 replay buffer 容量按二者较小值创建，因此当前最多保存 1,000,000 条 expert transition。

## PPO 接入点

### Wrapper

修改文件：

```text
scripts_attnnopriv/rsl_rl/local_rsl_rl/wrappers/VecEnvWrapper.py
```

新增 `_extract_amp_observation()`，并在结构化 observation dict 中加入：

```python
"amp_obs": self._extract_amp_observation()
```

### Storage

修改文件：

```text
scripts_attnnopriv/rsl_rl/local_rsl_rl/storage/cts_moe_rollout_storage.py
```

新增字段：

```python
amp_obs
next_amp_obs
task_rewards
amp_rewards
```

PPO mini-batch generator 不返回 AMP 字段。discriminator update 直接从 storage 中读取并 flatten。

### Algorithm

修改文件：

```text
scripts_attnnopriv/rsl_rl/local_rsl_rl/algorithms/cts_moe_ppo.py
```

新增逻辑：

1. 初始化 `AMPMotionDataset`、`AMPDiscriminator`、`amp_optimizer`。
2. `act(..., amp_obs=None)` 保存当前 AMP obs。
3. `process_env_step(..., next_amp_obs=None)` 计算 AMP reward 并存储 transition。
4. `update()` 中在 PPO 和 student distillation 之后更新 discriminator。

每轮 AMP discriminator 更新次数：

```text
amp_num_learning_epochs * amp_num_mini_batches = 5 * 4 = 20
```

当前 `amp_batch_size = 24576`，对应：

```text
4096 envs * 24 steps / 4 mini_batches
```

如果 rollout size 小于 `amp_batch_size`，代码会自动将 mini-batch size 降到 rollout size，便于小规模测试。

### Runner

修改文件：

```text
scripts_attnnopriv/rsl_rl/local_rsl_rl/runners/on_policy_runner.py
```

rollout 中将下一帧 AMP obs 传给 algorithm：

```python
self.alg.process_env_step(
    rewards,
    dones,
    infos,
    next_amp_obs=next_obs.get("amp_obs"),
)
```

hybrid runner 分支当前不启用 AMP，会过滤 `amp_obs` 后再传给 hybrid algorithm。

checkpoint 中新增保存：

```python
amp_discriminator_state_dict
amp_optimizer_state_dict
```

## 配置参数

已在 CTS-MoE algorithm config 中加入：

```python
use_amp = True
amp_motion_files = "datasets/mocap_motions/*"
amp_reward_coef = 0.01
amp_task_reward_lerp = 0.3
amp_discr_hidden_dims = [1024, 512]
amp_replay_buffer_size = 1_000_000
amp_num_preload_transitions = 2_000_000
amp_num_learning_epochs = 5
amp_num_mini_batches = 4
amp_batch_size = 24_576
amp_discriminator_lr = 1.0e-3
amp_grad_penalty_lambda = 10.0
amp_policy_target = -1.0
amp_expert_target = 1.0
amp_obs_dim = 30
amp_discriminator_input_dim = 60
```

已补充的配置文件包括：

```text
source/Go2Piper_Attention/Go2Piper_Attention/tasks/manager_based/go2piper_attention/config/agents/rsl_rl_ppo_cfg_moe.py
source/Go2Piper_Attention/Go2Piper_Attention/tasks/manager_based/go2piper_attention/config/agents/rsl_rl_ppo_cfg_moe_ortho.py
source/Go2Piper_Attention/Go2Piper_Attention/tasks/manager_based/go2piper_attention/config/agents/rsl_rl_ppo_cfg_moe_ortho_CNN.py
source/Go2Piper_Attention/Go2Piper_Attention/tasks/manager_based/go2piper_attention/config/agents/rsl_rl_ppo_cfg_moe_linear_gate.py
source/Go2Piper_Attention/Go2Piper_Attention/tasks/manager_based/go2piper_attention/config/agents/rsl_rl_ppo_cfg_moe_no_ortho.py
```

`rsl_rl_ppo_cfg_moe_ortho_separate.py` 对应 `HybridLegArmPPO`，当前没有接 AMP 参数。

## 日志

新增 AMP 日志 key：

```text
AMP/reward_mean
AMP/task_reward_mean
AMP/total_reward_mean
AMP/discriminator
AMP/expert_loss
AMP/policy_loss
AMP/grad_penalty
AMP/expert_d_mean
AMP/policy_d_mean
AMP/expert_acc
AMP/policy_acc
```

这些 key 会直接写入 logger，不再额外加 `Loss/` 前缀。

## 验证状态

已完成：

```text
python3 -m py_compile
```

覆盖文件：

```text
amp_discriminator.py
amp_motion_dataset.py
cts_moe_rollout_storage.py
cts_moe_ppo.py
on_policy_runner.py
VecEnvWrapper.py
相关 CTS-MoE agent cfg 文件
```

未完成：

```text
AMP tensor smoke test
```

原因：当前裸 `python3` 环境中没有 `torch` 模块。需要在 Isaac/RSL-RL 训练环境中再运行一次实际 smoke test 或短训练。

## 建议检查点

第一次启动训练时建议重点观察：

```text
AMP/reward_mean
AMP/expert_d_mean
AMP/policy_d_mean
AMP/expert_acc
AMP/policy_acc
AMP/grad_penalty
```

如果 `AMP/policy_d_mean` 很快贴近 `-1`，说明 discriminator 相对 policy 太强，可以考虑降低 `amp_discriminator_lr` 或减少 `amp_num_learning_epochs`。

如果 `AMP/reward_mean` 长期接近 0，优先检查 mocap 与仿真侧的 `joint_pos` 参考系是否一致。
