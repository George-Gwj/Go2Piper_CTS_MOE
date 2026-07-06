# Structure-aware CTS-MoE 模块说明

本目录新增了一个面向 Go2Piper 四足机械臂的 Structure-aware CTS-MoE PyTorch 网络模块。当前实现只包含网络结构，不改 PPO、runner、rollout storage 或 IsaacLab 环境主逻辑。

新增模块包括：

- `TeacherEncoder`：训练阶段使用的 teacher 编码器，输入 height scan、privileged vector 和可选 context。
- `StudentEncoder`：部署阶段使用的 student 编码器，输入本体历史和 raycaster depth image。
- `MoEActor`：dense soft routing 的 Mixture-of-Experts actor。
- `SparseMultiCritic`：按任务选择 value head 的 sparse multi-critic。
- `StructureAwareCTSMoEPolicy`：总封装类，组合 teacher/student encoder、MoE actor、multi-critic 和可学习 Gaussian `log_std`。

实现文件：

```text
scripts_attnnopriv/rsl_rl/local_rsl_rl/modules/structure_aware_cts_moe.py
```

导出入口：

```text
scripts_attnnopriv/rsl_rl/local_rsl_rl/modules/__init__.py
```

配置字段位置：

```text
source/Go2Piper_Attention/Go2Piper_Attention/tasks/manager_based/go2piper_attention/config/agents/rsl_rl_ppo_cfg_moe.py
```

## 文档同步约定

之后每次修改 `structure_aware_cts_moe.py`、CTS-MoE 相关配置、接口输入输出、网络维度或接入边界时，都需要同步更新本文档，避免 README 和代码行为不一致。

## Height Scan 输入约定

`TeacherEncoder` 期望 multi-layer height scan 以 `[B, 3, H, W]` 的形式输入：

```python
height_scan.shape == [B, 3, H, W]
height_scan[:, 0]  # H_ground_scan
height_scan[:, 1]  # H_lateral_scan
height_scan[:, 2]  # H_overhead_scan
```

这三个 channel 对应 `go2piper_cts_moe_env_cfg.py` 里的三个 scan sensor：

- `H_ground_scan`
- `H_lateral_scan`
- `H_overhead_scan`

CTS-MoE 环境配置中新增了独立 observation group：

```python
observations.height_scan
```

该 group 使用 `mdp.cts_moe_height_scan()`，将三层 ray caster 输出 stack 成：

```python
height_scan.shape == [B, 3, num_rays, 1]
```

这里的 `num_rays` 来自 `patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0])`。由于 teacher 当前使用 MLP 并直接 flatten height map，最后一维设为 `1` 不影响编码，只是保留 `[B, C, H, W]` 的接口约定。

当前实现会将三层 height scan flatten 成 `h_t`，再送入 Heightmap MLP：

```text
MLP(h_t): [512, 256] -> 128
```

实现上 `h_t` 是 `height_scan.flatten(start_dim=1)`。如果实例化 `TeacherEncoder` 时传入 `height_flat_dim`，第一层使用固定 `Linear`；如果 `height_flat_dim=None`，第一层使用 `LazyLinear`，在第一次前向时根据实际 height scan 分辨率自动初始化。

Privileged vector 走独立 MLP：

```text
MLP(e_t): [512, 256] -> 32
```

最后将 `c_t ⊕ MLP(e_t) ⊕ MLP(h_t)` 输入 `Linear + LayerNorm`，输出 32 维 teacher latent。`c_t` 是可选 context；当 `context_dim=0` 时不使用。

`TeacherEncoder` 还包含一个 task classifier：

```python
task_logits = task_classifier(z_teacher)
teacher_task_id = torch.argmax(task_logits, dim=-1)
```

默认调用仍然只返回 `z_teacher`：

```python
z_teacher = teacher_encoder(height_scan, privileged_obs)
```

需要 task 输出时可以显式请求：

```python
z_teacher, teacher_task_id = teacher_encoder(
    height_scan,
    privileged_obs,
    return_task_id=True,
)

z_teacher, teacher_task_id, teacher_task_logits = teacher_encoder(
    height_scan,
    privileged_obs,
    return_task_id=True,
    return_task_logits=True,
)
```

`teacher_task_id` 是 `[B]` long tensor，`teacher_task_logits` 是 `[B, num_tasks]`。

`TeacherEncoder` 当前默认参数：

```python
latent_dim = 32
height_channels = 3
num_tasks = 4
context_dim = 0
height_feature_dim = 128
privileged_feature_dim = 32
height_hidden_dims = (512, 256)
privileged_hidden_dims = (512, 256)
```

## Student Depth 输入约定

`StudentEncoder` 使用 `depth_image` 对应的 raycaster camera 图像作为 `d_t`：

```python
depth.shape == [B, C, 58, 87]
```

CTS-MoE 环境配置中新增了独立 observation group：

```python
observations.depth
```

该 group 使用 `mdp.cts_moe_depth_image()` 读取 `depth_image` sensor 的 `distance_to_image_plane`，并转为 CNN 需要的 channel-first 格式：

```python
depth.shape == [B, 1, 58, 87]
```

`observations.depth` 和 `observations.height_scan` 都设置为 `concatenate_terms=False`，目的是保持图像/height tensor 的结构，不把它们拼进 leg/arm proprio 的一维向量里。

Depth CNN 结构：

```text
filters: [16, 32, 64]
output: 128
```

Proprioception history 会 flatten 成 `o^p_{t-H:t}` 后进入 MLP。CTS-MoE 环境中当前约定：

```python
H = 5
```

也就是 `go2piper_cts_moe_env_cfg.py` 中本体相关 history observation 使用 `history_length=5`。Depth perception 不使用历史帧，只读取当前帧 `depth_image`。

CTS-MoE 环境已经将旧的 leg/arm observation group 合并成 full-body group：

```python
observations.proprio          # 当前帧 full-body proprio，供 actor / critic 使用
observations.proprio_history  # H=5 full-body proprio history，供 StudentEncoder 使用
observations.privileged_obs   # teacher / critic privileged vector
```

`observations.proprio` 和 `observations.proprio_history` 都拼接 leg 与 arm 的本体信息，并且包含 command：

```text
base_ang_vel
leg_joint_pos, leg_joint_vel, leg_actions
arm_joint_pos, arm_joint_vel, arm_actions
base_velocity command
ee_pose command
projected_gravity
```

当前 `proprio_dim = 66`：

```text
3 + 12 + 12 + 12 + 6 + 6 + 6 + 3 + 3 + 3 = 66
```

`observations.privileged_obs` 在 full-body proprio 基础上额外包含：

```text
base_lin_vel
leg_joint_torques
arm_joint_torques
feet_contact
gripper_pose
```

当前 `privileged_dim = 98`：

```text
3 + 3 + 12 + 12 + 12 + 6 + 6 + 6 + 3 + 3 + 3 + 12 + 6 + 4 + 7 = 98
```

旧的 `leg_policy`、`arm_policy`、`leg_critic`、`arm_critic` group 已不再在 CTS-MoE 环境中注册。

```text
MLP(o^p_{t-H:t}): [512, 256] -> 32
```

Temporal GRU 输入为 `phi(d_t) ⊕ o^p_t`：

```text
GRU input: 128 + proprio_dim
GRU output: 256
layers: 1
```

Student latent projection 为 `Linear + LayerNorm`：

```text
GRU(.) ⊕ MLP(o^p_{t-H:t}) -> 32
```

`StudentEncoder.forward(proprio_history, perception)` 不再单独接收 `cmd`。当前约定是 `proprio_dim` 对应的 proprioception 向量已经包含 command 信息。

`StudentEncoder` 当前默认参数：

```python
latent_dim = 32
perception_type = "depth"
perception_channels = 1
proprio_feature_dim = 32
proprio_hidden_dims = (512, 256)
depth_feature_dim = 128
depth_filters = (16, 32, 64)
gru_hidden_dim = 256
gru_num_layers = 1
```

## MoE 配置文件

`rsl_rl_ppo_cfg_moe.py` 已从旧的 `leg_policy + arm_policy` 双策略配置，收敛为 single full-body CTS-MoE 配置：

```python
Go2PiperCTSMoEPolicyCfg
Go2PiperCTSMoEAlgorithmCfg
Go2PiperCTSMoERunnerCfg
```

已经删除的旧字段包括：

- `leg_policy`
- `arm_policy`
- `leg_algorithm`
- `arm_algorithm`
- `num_leg_actions`
- `num_arm_actions`
- 旧 RMA / attention / privileged encoder 遗留字段
- `NoPriv_Go2PiperFlatPPORunnerCfg`

`Go2PiperCTSMoEPolicyCfg` 集中配置 `StructureAwareCTSMoEPolicy` 的网络结构：

```python
class_name = "StructureAwareCTSMoEPolicy"

proprio_dim = 66           # full-body proprio，已包含 command
privileged_dim = 98        # full-body privileged obs
action_dim = 18
latent_dim = 32
num_tasks = 4

height_channels = 3
teacher_context_dim = 0
teacher_height_flat_dim = None
teacher_privileged_hidden_dims = [512, 256]
teacher_privileged_feature_dim = 32
teacher_height_hidden_dims = [512, 256]
teacher_height_feature_dim = 128

student_perception_type = "depth"
student_perception_channels = 1
student_proprio_hidden_dims = [512, 256]
student_proprio_feature_dim = 32
student_depth_filters = [16, 32, 64]
student_depth_feature_dim = 128
student_gru_hidden_dim = 256
student_gru_num_layers = 1

num_experts = 4
expert_hidden_dims = [256, 128]
router_hidden_dims = [128, 64]
expert_names = ["lateral_avoidance", "under_table", "stair_up", "stair_down"]

critic_hidden_dims = [256, 128]
critic_shared_trunk = False
critic_trunk_hidden_dims = None
critic_head_hidden_dims = [64]

init_log_std = 0.0
learnable_log_std = True
```

`Go2PiperCTSMoEAlgorithmCfg` 集中配置 `CTSMoEPPO`：

```python
class_name = "CTSMoEPPO"

learning_rate = 1e-3
student_learning_rate = 1e-4
num_learning_epochs = 5
num_mini_batches = 4
clip_param = 0.2
entropy_coef = 0.005
value_loss_coef = 1.0
schedule = "adaptive"
desired_kl = 0.01

distillation_loss_coef = 1.0
student_rollout_ratio = 0.15
router_entropy_coef = 0.0
router_balance_coef = 0.0
router_logit_l2_coef = 0.0
per_task_advantage_normalization = True

use_popart = False
popart_beta = 0.99999
popart_eps = 1e-5
popart_min_std = 1e-2
popart_use_output_rescale = True
popart_value_loss = "huber"
popart_huber_delta = 1.0
value_loss_per_task_average = True
```

`student_rollout_ratio` 控制 rollout 数据收集阶段 student latent 的 env 占比：

- 默认 `student_rollout_ratio = 0.15`，即 teacher:student = 85:15。
- `student_mask=True` 的 env 使用 `z_student`。
- `student_mask=False` 的 env 使用 `z_teacher`。
- 如果 runner 显式传入 `student_mask`，则使用外部 mask；如果不传，`CTSMoEPPO.act()` 会按该比例随机生成。

`Go2PiperCTSMoERunnerCfg` 当前只提供配置入口。真正从环境训练还需要后续 runner 将 flat observation group 映射成 CTS-MoE structured tensors。

## 主要接口

Teacher 路径：

```python
z_teacher = policy.encode_teacher(height_scan, privileged_obs)
z_teacher, teacher_task_id, teacher_task_logits = policy.encode_teacher(
    height_scan,
    privileged_obs,
    return_task_id=True,
    return_task_logits=True,
)
action_mean, router_weights, expert_actions, router_logits = policy.act_teacher(
    height_scan, privileged_obs, proprio
)
value = policy.evaluate_teacher(height_scan, privileged_obs, proprio, task_id)
```

Student 路径：

```python
z_student = policy.encode_student(proprio_history, perception)
action_mean, router_weights, expert_actions, router_logits = policy.act_student(
    proprio_history, perception, proprio
)
value = policy.evaluate_student(proprio_history, perception, proprio, task_id)
```

Mixed 路径：

```python
out = policy(
    mode="mixed",
    height_scan=height_scan,
    privileged_obs=privileged_obs,
    proprio_history=proprio_history,
    perception=depth,
    proprio=proprio,
    student_mask=student_mask,
)
```

`student_mask` 必须是 `[B]` bool tensor：

- `True`：该样本使用 `z_student`。
- `False`：该样本使用 `z_teacher`。

mixed latent 的选择逻辑：

```python
z_teacher = encode_teacher(height_scan, privileged_obs)
z_student = encode_student(proprio_history, perception)
z = torch.where(student_mask[:, None], z_student, z_teacher)
```

如果 `detach_student_in_mixed=True`，则使用：

```python
z = torch.where(student_mask[:, None], z_student.detach(), z_teacher)
```

`forward()` 不会默认 detach student latent，是否 detach 由调用方或 `detach_student_in_mixed` 显式决定。

当 `forward(mode="teacher")` 或 `forward(mode="mixed")` 时，输出 dict 会额外包含：

```python
out["teacher_task_id"]      # [B], long
out["teacher_task_logits"]  # [B, num_tasks]
```

`mode="student"` 不计算 teacher encoder，因此不会包含这两个字段。

Teacher-student latent 对齐：

```python
loss = policy.distillation_loss(z_student, z_teacher)
```

`distillation_loss` 固定使用：

```python
loss = F.mse_loss(z_student, z_teacher.detach())
```

返回值是 scalar tensor。

## MoE Actor 输出

`MoEActor.forward(z, proprio)` 返回四个对象：

```python
action_mean, router_weights, expert_actions, router_logits = moe_actor(z, proprio)
```

shape 约定：

```text
action_mean:    [B, action_dim]
router_weights: [B, num_experts]
expert_actions: [B, num_experts, action_dim]
router_logits:  [B, num_experts]
```

`router_logits` 用于后续 PPO 辅助项，例如 router entropy、router balance、router logit L2。Actor/router 不接收 `task_id`。

Router 和 expert 的输入不同：

```python
router_logits = router(z)
expert_actions = experts(torch.cat([z, proprio], dim=-1))
```

也就是说 router 的 expert 权重完全由 latent `z` 决定；`proprio` 只进入各个 expert 的动作输出网络，不进入 router。当前约定中 command 已经包含在 `proprio` 内，不再作为单独 `cmd` 输入。

## Action Distribution Helper

`StructureAwareCTSMoEPolicy` 提供：

```python
dist = policy.get_action_distribution(action_mean, action_std)
```

内部返回：

```python
torch.distributions.Normal(action_mean, action_std)
```

`forward()` 输出 dict 中也会包含：

```python
out["action_mean"]
out["action_std"]
out["distribution"]
```

后续 PPO 可以直接使用：

```python
log_prob = dist.log_prob(actions).sum(dim=-1)
entropy = dist.entropy().sum(dim=-1)
```

## Expert 和 Task 语义

默认 actor experts：

- `lateral_avoidance`
- `under_table`
- `stair_up`
- `stair_down`

默认 critic tasks：

- `0`：box avoidance
- `1`：under-table
- `2`：stair-up
- `3`：stair-down

Actor/router 不接收 `task_id`，避免把任务标签直接喂给策略路由器。`task_id` 只在 critic 中使用，用于为 batch 内每个样本选择对应的 value head。

## Sparse Multi-Critic

`SparseMultiCritic` 支持两种结构：

默认 `critic_shared_trunk=False`：

```text
每个 task 一个完整独立 critic MLP
input:  [z, proprio]
hidden: critic_hidden_dims
output: 1
```

可选 `critic_shared_trunk=True`：

```text
shared trunk + task-specific value heads
```

`task_id` 只用于选择 critic head：

```python
value = all_values[batch_idx, task_id]
```

batch 内不同 `task_id` 可以选择不同 value head。`task_id` 不会进入 actor 或 router。

## PPO 参数分组

`StructureAwareCTSMoEPolicy` 提供两个参数分组方法，方便后续使用两个 optimizer：

```python
ppo_optimizer = Adam(policy.ppo_parameters())
student_optimizer = Adam(policy.student_parameters())
```

`ppo_parameters()` 包含：

- `teacher_encoder`
- `moe_actor`
- `multi_critic`
- `log_std`，仅当它是 `nn.Parameter` 时包含

`student_parameters()` 只包含：

- `student_encoder`

这样 PPO 更新不会直接更新 `student_encoder`，student encoder 后续由 distillation loss 单独训练。

## CTS-MoE PPO

新增专用 PPO 和 rollout storage：

```text
scripts_attnnopriv/rsl_rl/local_rsl_rl/algorithms/cts_moe_ppo.py
scripts_attnnopriv/rsl_rl/local_rsl_rl/storage/cts_moe_rollout_storage.py
```

导出入口：

```python
from local_rsl_rl.algorithms import CTSMoEPPO
from local_rsl_rl.storage import CTSMoERolloutStorage
```

`CTSMoEPPO` 不替换旧的 `PPO` 类，避免影响原来的 leg/arm 双 policy 训练路径。

训练逻辑分为两个阶段：

阶段 1：PPO update。

- 只更新 `policy.ppo_parameters()`：
  - `teacher_encoder`
  - `moe_actor`
  - `multi_critic`
  - `log_std`，仅当它是 `nn.Parameter`
- PPO loss 不更新 `student_encoder`。
- mixed mode 中强制 `detach_student_in_mixed=True`，避免 PPO surrogate/value loss 反向传播到 `student_encoder`。

阶段 2：PPO update 完成之后，再单独做 student distillation。

- 只更新 `policy.student_parameters()`。
- 只使用 `student_mask=True` 的 student 轨迹样本。
- 不使用 teacher-only 轨迹做 distillation。

```python
student_traj = student_mask == True
loss_distill = policy.distillation_loss(
    z_student[student_traj],
    z_teacher[student_traj],
)
```

其中 `distillation_loss` 内部使用 `z_teacher.detach()`。

Rollout 和 PPO update 阶段都使用 mixed mode：

```python
out = policy(
    mode="mixed",
    proprio=proprio,
    task_id=task_id,
    height_scan=height_scan,
    privileged_obs=privileged_obs,
    proprio_history=proprio_history,
    perception=perception,
    student_mask=student_mask,
    detach_student_in_mixed=True,
    return_value=True,
)
```

`detach_student_in_mixed=True` 是 PPO 阶段的强制要求；distillation 阶段不走 mixed actor，而是直接计算 `encode_teacher()` 和 `encode_student()`。

`CTSMoEPPO` 内部维护两个 optimizer：

```python
ppo_optimizer = Adam(policy.ppo_parameters())
student_optimizer = Adam(policy.student_parameters())
```

阶段 1 的 PPO loss：

```text
ppo_loss =
    surrogate_loss
  + value_loss_coef * sparse_critic_value_loss
  - entropy_coef * action_entropy
  + router_aux_loss
```

阶段 2 的 student loss：

```text
student_loss =
    distillation_loss_coef * MSE(
        z_student[student_mask],
        z_teacher[student_mask].detach()
    )
```

`router_aux_loss` 是可选项：

```text
router_aux_loss =
  - router_entropy_coef * router_entropy
  + router_balance_coef * router_balance_loss
  + router_logit_l2_coef * router_logit_l2_loss
```

默认所有 router aux 系数为 0。

## Per-Task Advantage Normalization

`CTSMoERolloutStorage.compute_returns()` 支持 per-task advantage normalization，默认开启：

```python
per_task_advantage_normalization = True
```

归一化逻辑是按 `task_id` 分组，对每个 task 内部的 advantage 单独计算 mean/std：

```python
for task in unique(task_id):
    advantage[task] = (advantage[task] - mean_task) / (std_task + eps)
```

这样不会把 box avoidance、under-table、stair-up、stair-down 的 advantage 混在一起归一化。

## Per-Task POPArt Return Normalization

新增实现：

```text
scripts_attnnopriv/rsl_rl/local_rsl_rl/utils/popart.py
```

导出入口：

```python
from local_rsl_rl.utils import PerTaskPopArt
```

`PerTaskPopArt` 为每个 task 维护独立 return statistics：

```python
mean:          [num_tasks]
std:           [num_tasks]
second_moment: [num_tasks]
```

主要接口：

```python
old_stats, new_stats = popart.update(task_ids, returns_raw)
returns_norm = popart.normalize(task_ids, returns_raw)
values_raw = popart.denormalize(task_ids, values_norm)
stats = popart.get_stats()
```

默认参数：

```python
popart_beta = 0.99999
popart_eps = 1e-5
popart_min_std = 1e-2
popart_use_output_rescale = True
popart_value_loss = "huber"
popart_huber_delta = 1.0
value_loss_per_task_average = True
```

启用 `use_popart=True` 后，critic 输出语义变为 normalized value：

```text
policy.multi_critic(...) -> V_norm
```

但 storage 始终保存 raw scale：

```text
storage.values  = V_raw
storage.returns = R_raw
```

rollout `act()` 中：

```python
value_norm = out["value"]
value_raw = popart.denormalize(task_id, value_norm)
transition.values = value_raw.detach()
```

`compute_returns()` 中：

```python
last_value_norm = out["value"]
last_value_raw = popart.denormalize(task_id, last_value_norm)
storage.compute_returns(last_value_raw, ...)
```

因此 GAE 始终使用 raw rewards、raw values 和 raw returns。

PPO update 开始前，每个 rollout 更新一次 POPArt statistics：

```python
old_stats, new_stats = popart.update(storage.task_ids, storage.returns)
```

如果 `popart_use_output_rescale=True`，会对每个 task critic head 的最后一层 Linear 做 POPArt rescale，使 denormalized prediction 在统计量更新前后尽量保持不变：

```text
w_new = old_std / new_std * w_old
b_new = (old_std * b_old + old_mean - new_mean) / new_std
```

实现中使用与 normalize/denormalize 一致的 effective std：`std + eps`。

启用 POPArt 时，value loss 在 normalized space 中计算：

```python
R_norm = popart.normalize(task_id, returns_raw)
V_norm = out["value"]
old_V_norm = popart.normalize(task_id, old_values_raw)
```

支持 clipped value loss：

```python
V_clipped_norm = old_V_norm + (V_norm - old_V_norm).clamp(-clip, clip)
```

支持 Huber 或 MSE：

```python
popart_value_loss = "huber"  # or "mse"
```

默认 `value_loss_per_task_average=True`，即先对 batch 中每个 task 内部 value loss 求均值，再对出现过的 task 求均值。

POPArt 只影响 value output/value loss，不改变：

- actor surrogate loss
- rewards
- raw GAE returns
- per-task advantage normalization
- student distillation loss
- actor/router 输入

启用 POPArt 时，`update()` 返回 dict 额外包含：

```text
value_loss_norm
returns_norm_mean
returns_norm_std
popart_mean_task_0
popart_std_task_0
...
```

`use_popart=False` 时保持原来的 raw value loss 逻辑。

## Shape Check

当前已添加基础 shape 检查：

- `MoEActor.forward`
  - `z: [B, latent_dim]`
  - `proprio: [B, proprio_dim]`
  - batch size 必须一致
- `SparseMultiCritic.forward`
  - `z: [B, latent_dim]`
  - `proprio: [B, proprio_dim]`
  - `task_id: [B]`
  - `task_id` 范围必须在 `[0, num_tasks - 1]`
- `StructureAwareCTSMoEPolicy.forward`
  - `mode` 必须是 `"teacher"`、`"student"` 或 `"mixed"`
  - 不同 mode 的必需输入必须存在
  - mixed mode 必须提供 `[B]` bool `student_mask`

## 当前接入边界

当前已经新增网络模块、CTS-MoE 专用 PPO、CTS-MoE 专用 rollout storage、CTS-MoE observation groups、MoE task config 和 Gym task 注册。

Gym task：

```text
Go2Piper-Attention-CTS-MoE
```

该 task 使用：

```text
env_cfg_entry_point = config.moe_env_cfg:Go2PiperMoEEnvCfg
rsl_rl_cfg_entry_point = config.agents.rsl_rl_ppo_cfg_moe:Go2PiperCTSMoERunnerCfg
```

`Go2PiperMoEEnvCfg` 继承 `go2piper_cts_moe_env_cfg.LocomotionVelocityEnvCfg`，并保留 flat terrain、command curriculum 和 reward 权重设置。

仍未修改：

- legacy `PPO` 和 legacy `RolloutStorage` 的行为。
- `OnPolicyRunner` 主训练循环。
- reward 或 termination 逻辑。

后续要真正从环境训练，需要在 runner 中把 IsaacLab observation dict 映射成 `CTSMoEPPO` 需要的结构化输入：

- `proprio`
- `height_scan`
- `privileged_obs`
- `proprio_history`
- `perception`
- `task_id`
- `student_mask`

## 多任务 Env 接口

CTS-MoE 环境在 `ManagerRLEnv` 中预留了固定 env-task 分配和多任务 reward dispatcher。

任务常量：

```python
TASK_BOX_AVOIDANCE = 0
TASK_UNDER_TABLE = 1
TASK_STAIR_UP = 2
TASK_STAIR_DOWN = 3
NUM_TASKS = 4
TASK_NAMES = ("box_avoidance", "under_table", "stair_up", "stair_down")
```

配置位于 `go2piper_cts_moe_env_cfg.py`：

```python
multi_task_rewards = MultiTaskRewardCfg(
    alive_weight=0.1,
    fixed_task_assignment=True,
    fixed_task_id=None,
    task_sampling_weights=None,
)
```

默认 `fixed_task_assignment=True`。环境创建后调用 `_assign_env_tasks()`，按 env id 连续、均衡分配 task：

```text
env 0..N0-1 -> box_avoidance
env N0..N1-1 -> under_table
...
```

reset 时不会重新采样 task。只有当 `fixed_task_assignment=False` 时，`_reset_idx()` 才会调用 `_sample_task_ids(env_ids)`。`fixed_task_id` 可用于单任务 debug。

场景接口：

```python
_setup_task_scenes()
_reset_task_scene(env_ids)
```

当前第一版只预留 mask 和 TODO，不创建复杂场景元素：

```python
mask_box
mask_under_table
mask_stair_up
mask_stair_down
```

reward dispatcher：

```python
_get_rewards()
_reward_common()
_reward_box_avoidance()
_reward_under_table()
_reward_stair_up()
_reward_stair_down()
```

当前 reward dispatcher 按 `RewardsCfg` 中 reward term 的名称后缀自动分类：

```text
*_common        -> _reward_common()
*_box_avoidance -> _reward_box_avoidance()
*_under_table  -> _reward_under_table()
*_stair_up     -> _reward_stair_up()
*_stair_down   -> _reward_stair_down()
```

对应分类逻辑在 `RewardManager.compute_grouped_by_task_marker()` 中。

当前 common reward 包含两部分：

1. `RewardsCfg` 中所有以 `_common` 结尾的 full-body 基础奖励。
   这些权重主要在 `config/moe_env_cfg.py` 的 common reward block 中设置，例如 arm tracking、leg velocity tracking、orientation、action rate、smoothness、torque、feet contact 等。

2. CTS-MoE 额外预留的 alive reward：

```python
grouped_rewards, grouped_logs = reward_manager.compute_grouped_by_task_marker(dt=step_dt)
logs["common/alive"] = alive_weight
```

common reward logs 包括：

```python
logs["common/<clean_reward_name>"]
logs["common/marked_total"]
logs["common/alive"]
```

四个 task-specific reward 暂时都有 placeholder。后续针对不同任务设计奖励时，在 `RewardsCfg` 中新增带对应后缀的 `RewTerm`，然后在 `config/moe_env_cfg.py` 的对应 block 中设置权重：

```text
xxx_box_avoidance
xxx_under_table
xxx_stair_up
xxx_stair_down
```

这些 reward 会自动进入 `_reward_box_avoidance()`、`_reward_under_table()`、`_reward_stair_up()`、`_reward_stair_down()`，并只通过 task mask 加到对应 task env。

extras 输出：

```python
extras["task_id"]          # [num_envs], torch.long
extras["task_names"]       # tuple[str, ...]
extras["log"]["rew/common/alive"]
extras["log"]["rew/common/marked_total"]
extras["log"]["rew/box/placeholder"]
extras["log"]["rew/under_table/placeholder"]
extras["log"]["rew/stair_up/placeholder"]
extras["log"]["rew/stair_down/placeholder"]
extras["log"]["task/num_box"]
extras["log"]["task/num_under_table"]
extras["log"]["task/num_stair_up"]
extras["log"]["task/num_stair_down"]
```

`task_id` 也会作为 observation dict 的顶层 key 暴露：

```python
obs["task_id"] = env.task_id
```

它不拼进 `observations.proprio`，因此不会自动进入 actor/router。Runner 需要显式读取该 key 或 `extras["task_id"]`，并只传给 critic/storage/POPArt/advantage normalization。

progress buffer 已预留：

```python
prev_base_pos: [num_envs, 3]
_update_reward_buffers()
```

`prev_base_pos` 在 reward 计算后更新，避免后续 progress reward 在计算前被覆盖为当前位姿。

dry-run：

```bash
conda run --no-capture-output -n env_isaaclab_2026 \
  python scripts_attnnopriv/rsl_rl/tests/test_cts_moe_multitask_env_interface.py
```

## CTS-MoE 训练接入

当前 `scripts_attnnopriv/rsl_rl/local_rsl_rl/runners/on_policy_runner.py` 已切换为 CTS-MoE-only runner。

旧的 `leg`、`arm`、`all` 双 policy 训练分支已经移除。Runner 现在只接受：

```python
cfg["policy"]      # StructureAwareCTSMoEPolicy
cfg["algorithm"]   # CTSMoEPPO
```

初始化流程：

1. 从 env wrapper 读取 structured observation。
2. 用实际 observation shape 覆盖 cfg 中可能过期的 `proprio_dim`、`privileged_dim`、`action_dim`、`height_channels`、`student_perception_channels`。
3. 创建单一 `StructureAwareCTSMoEPolicy`。
4. 创建单一 `CTSMoEPPO`。
5. 用结构化 observation shape 初始化 `CTSMoERolloutStorage`。

训练 rollout 中使用：

```python
obs = env.get_cts_moe_observations()
actions = alg.act(
    proprio=obs["proprio"],
    height_scan=obs["height_scan"],
    privileged_obs=obs["privileged_obs"],
    proprio_history=obs["proprio_history"],
    perception=obs["perception"],
    task_id=obs["task_id"],
)
next_obs, rewards, dones, infos = env.step_cts_moe(actions)
alg.process_env_step(rewards, dones, infos)
```

计算 returns 时同样显式传入 `task_id`：

```python
alg.compute_returns(
    proprio=obs["proprio"],
    height_scan=obs["height_scan"],
    privileged_obs=obs["privileged_obs"],
    proprio_history=obs["proprio_history"],
    perception=obs["perception"],
    task_id=obs["task_id"],
)
```

因此 `task_id` 会进入 storage、SparseMultiCritic、per-task advantage normalization 和 POPArt，但不会拼进 `proprio`，也不会作为 actor/router 输入。

`scripts_attnnopriv/rsl_rl/local_rsl_rl/wrappers/VecEnvWrapper.py` 也已经切换为 CTS-MoE-only wrapper。它要求 observation manager 中存在以下 group：

```text
proprio
proprio_history
privileged_obs
height_scan
depth
```

wrapper 输出给 runner 的 observation dict：

```python
{
    "proprio": ...,
    "proprio_history": ...,  # [num_envs, H, proprio_dim]
    "privileged_obs": ...,
    "height_scan": ...,
    "perception": ...,
    "task_id": ...,          # [num_envs], torch.long
}
```

其中 `depth` group 会映射为 policy 侧的 `perception`，`height_scan` group 会保持为 teacher heightmap 输入。

`scripts_attnnopriv/rsl_rl/train.py` 也已清理旧的 leg/arm checkpoint 逻辑。现在 resume 只加载单一 CTS-MoE checkpoint：

```python
runner.load(resume_path)
```

训练命令：

```bash
conda run --no-capture-output -n env_isaaclab_2026 \
  python scripts_attnnopriv/rsl_rl/train.py \
  --task Go2Piper-Attention-CTS-MoE
```

### 2026-07-06 Smoke Test 修复记录

已用 16 个 env 跑通 1 iteration smoke test：

```bash
conda run --no-capture-output -n env_isaaclab_2026 \
  python scripts_attnnopriv/rsl_rl/train.py \
  --task Go2Piper-Attention-CTS-MoE \
  --num_envs 16 \
  --max_iterations 1
```

本次 smoke test 暴露并修复了以下接入问题：

- 当前 `go2_piper_camera.usd` 暴露的腿部关节只有 `RR_*` 和 `RL_*`，没有 `FR_*` / `FL_*`。因此 CTS-MoE smoke 版本把腿部 action、leg observation 和相关 torque/acc reward 的关节列表对齐为 rear-leg-only，再与 6 个 arm joint 组成当前 full-body action。
- `GO2PIPER_CFG.init_state.joint_pos` 删除了不存在的 `F[L,R]_thigh_joint` 正则，避免 IsaacLab asset 初始化阶段报 “Not all regular expressions are matched”。
- `depth_image` 的 `prim_path` 临时从 `{ENV_REGEX_NS}/Robot/base/d435_camera` 改为 `{ENV_REGEX_NS}/Robot/base`。当前 USD 中没有 `d435_camera` prim；后续如果 USD 补回真实相机 prim，可以再切回。
- `F_joint_deviation_common` 临时改为 rear-leg joint 正则，避免 reward manager 初始化解析不存在的 front-leg joints。
- `RslRlVecEnvWrapper` 不再假设 `ProprioHistoryCfg` group 自身有 `history_length` 字段；如果 group-level 值为 `None`，默认使用 `H=5`。注意实际 history 是配置在每个 `ObsTerm(history_length=5)` 上。
- `OnPolicyRunner` 会过滤 IsaacLab/RSL-RL base PPO cfg 中 `CTSMoEPPO` 不支持的额外字段，例如 `normalize_advantage_per_mini_batch`。

smoke test 已确认：

```text
Total timesteps: 384
task/num_box: 4
task/num_under_table: 4
task/num_stair_up: 4
task/num_stair_down: 4
```

## Smoke Test 目标

建议用 synthetic tensor 先验证以下内容：

- `TeacherEncoder` 输出 `[B, latent_dim]`。
- `StudentEncoder` 支持 raycaster depth/grid perception；`vector` perception 仍保留为调试兼容路径。
- `MoEActor` 输出 `action_mean`、`router_weights`、`expert_actions`、`router_logits`，且 router 权重按 expert 维度求和为 1。
- `SparseMultiCritic` 支持 batch 内不同 `task_id` 选择不同 value head。
- `StructureAwareCTSMoEPolicy` 支持 `teacher/student/mixed` 三种 mode。
- `ppo_parameters()` 不包含 `student_encoder`，`student_parameters()` 只包含 `student_encoder`。
- `get_action_distribution()` 返回可用于 PPO log-prob 和 entropy 的 Normal distribution。
- `distillation_loss(z_student, z_teacher)` 可以正常反向传播。
- `CTSMoEPPO` 支持 mixed PPO、student distillation、sparse critic、per-task advantage normalization 和可选 router auxiliary loss。
- `PerTaskPopArt.normalize/denormalize` 能保持数值往返一致。
- `use_popart=True` 时，storage 存 raw value，value loss 使用 normalized prediction 和 normalized returns。
- `use_popart=False` 时，保持原 raw value loss 逻辑。
