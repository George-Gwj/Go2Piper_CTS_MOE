# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
import math
from typing import List
from torch import Tensor
from local_rsl_rl.utils import resolve_nn_activation
from local_rsl_rl.networks import EmpiricalNormalization


class ActorMLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dims: list[int],
                 output_dim: int = 12,
                 activation: nn.Module = nn.ELU(),
                 output_activation: nn.Module = nn.Identity()):   
        super().__init__()
        layers = []
        prev = input_dim
        # 隐藏层
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), activation]
            prev = h
        # 输出层
        layers += [nn.Linear(prev, output_dim), output_activation]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self,
                 obs_feature, 
                 attn_feature,
                 out_dim = 32,
                 num_heads: int = 4,
                 hidden_dims: list[int] = [256, 128],
                 output_dim: int = 12,
                 activation: nn.Module = nn.ELU(),
                 output_activation: nn.Module = nn.Identity()):
        super().__init__()

        self.obs_feature = obs_feature
        self.attn_feature = attn_feature

        self.fcs = nn.ModuleList(nn.Linear(d, out_dim) for d in obs_feature)
        self.attn_fcs = nn.ModuleList(nn.Linear(d, out_dim) for d in attn_feature)

        # 跨模态注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=out_dim, 
            num_heads=num_heads, 
            batch_first=True
        )

        # MLP网络
        layers = []
        current_dim = out_dim * len(obs_feature)
        for h in hidden_dims:
            layers += [nn.Linear(current_dim, h), activation]
            current_dim = h
        layers += [nn.Linear(current_dim, output_dim), output_activation]
        self.net = nn.Sequential(*layers)


    def forward(self, obs_sequence: Tensor, attn_sequence: Tensor):
        """
        obs_sequence:  [B, sum(obs_feature)]
        attn_sequence: [B, sum(attn_feature)]
        return:        (output [B, output_dim], attn_weights [B, num_heads, 6, 7])
        """

        B = obs_sequence.size(0)

        # 1. 先把每个段 pull 出来并立刻压成定长列表 → TorchScript 能看到长度
        obs_splits: List[Tensor] = []
        attn_splits: List[Tensor] = []
        start = 0
        for d in self.obs_feature:
            obs_splits.append(obs_sequence[:, start:start+d])
            start += d
        start = 0
        for d in self.attn_feature:
            attn_splits.append(attn_sequence[:, start:start+d])
            start += d

        # 2. 静态循环：长度 = ModuleList 长度，编译期可展开
        obs_tokens: List[Tensor] = []
        for i, fc in enumerate(self.fcs):
            token = fc(obs_splits[i]).unsqueeze(1)
            obs_tokens.append(token)
        obs_out = torch.cat(obs_tokens, dim=1)


        attn_tokens: List[Tensor] = []
        for i, fc in enumerate(self.attn_fcs):
            token = fc(attn_splits[i]).unsqueeze(1)
            attn_tokens.append(token)
        attn_out = torch.cat(attn_tokens, dim=1)

        # 3. 跨模态注意力
        attn_output, attn_weights = self.cross_attention(
            query=obs_out, key=attn_out, value=attn_out
        )                                               # attn_output: [B, N_obs, out_dim]

        # 4. 拉平 → MLP
        attn_flat = attn_output.reshape(B, -1)          # [B, N_obs * out_dim]

        output = self.net(attn_flat)                    # [B, output_dim]
        return output, attn_weights


class Actor_RMA(nn.Module):
    def __init__(self, 
                 mlp_input_dim_a, 
                 actor_hidden_dims, 
                 activation, 
                 control_head_hidden_dims,
                 num_actions,
                 obs_feature,
                 attn_feature,
                 actor_attn_hidden_dims,
                 attn_feature_output,
                 attn_output
                ):
        
        super().__init__()

        # attn latent 
        self.actor_attn = Attention(
            obs_feature=obs_feature,
            attn_feature=attn_feature,
            hidden_dims=actor_attn_hidden_dims,
            out_dim=attn_feature_output,  
            output_dim=attn_output,                
            activation=activation,
        )    

        # Actor backbone latent
        if len(actor_hidden_dims) > 0:
            actor_layers = []
            actor_layers.append(nn.Linear((mlp_input_dim_a + attn_output  ), actor_hidden_dims[0]))
            actor_layers.append(activation)
            for l in range(len(actor_hidden_dims) - 1):
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
            self.actor_backbone = nn.Sequential(*actor_layers)
            actor_backbone_output_dim = actor_hidden_dims[-1]
        else:
            self.actor_backbone = nn.Identity()
            actor_backbone_output_dim = mlp_input_dim_a + attn_output 

        # Actor control head latent
        actor_layers = []
        actor_layers.append(nn.Linear(actor_backbone_output_dim, control_head_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(control_head_hidden_dims)):
            if l == len(control_head_hidden_dims) - 1:
                actor_layers.append(nn.Linear(control_head_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(control_head_hidden_dims[l], control_head_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor_control_head = nn.Sequential(*actor_layers)

    # TODO： no hist and priv 
    def forward(self, obs,  obs_attn):
        latent = self.student_latent(obs, obs_attn)
        backbone_input = torch.cat([obs, latent], dim=1)
        backbone_output = self.actor_backbone(backbone_input)
        output = self.actor_control_head(backbone_output)
        return output
    
    def student_latent(self, obs, obs_attn):
        attn_output, _ =self.actor_attn(obs, obs_attn)
        return attn_output


class CriticBackbone(nn.Module):
    """
    Actor 主干网络：
      输入维度 = mlp_input_dim_a + priv_encoder_output_dim
      输出维度 = 最后一个隐藏层维度（无隐藏层时等于输入维度）
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dims: list[int],
                 activation: nn.Module = nn.ELU()):
        super().__init__()
        if len(hidden_dims) > 0:
            layers = []
            prev_dim = input_dim
            for h in hidden_dims:
                layers += [nn.Linear(prev_dim, h), activation]
                prev_dim = h
            self.net = nn.Sequential(*layers)
            self.output_dim = hidden_dims[-1]
        else:
            self.net = nn.Identity()
            self.output_dim = input_dim

    def forward(self, x):
        return self.net(x)


class CriticMLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dims: list[int],
                 output_dim: int,
                 activation: nn.Module = nn.ELU(),
                 output_activation: nn.Module = nn.Identity()):
        super().__init__()
        layers = []
        priv_dim = input_dim

        # 隐藏层
        for h in hidden_dims:
            layers += [nn.Linear(priv_dim, h), activation]
            priv_dim = h
        # 输出层
        layers += [nn.Linear(priv_dim, output_dim), output_activation]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,  
        num_actor_obs,
        num_critic_obs,
        obs_feature,
        attn_feature,
        critic_obs_feature,
        critic_attn_feature,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        activation='elu',
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            print(
                "Arm_ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )

        actor_attn_hidden_dims = kwargs['actor_attn_hidden_dims']
        self.min_std = kwargs['min_std']
        rma_actor_hidden_dims = kwargs['rma_actor_hidden_dims']
        rma_control_head_hidden_dims = kwargs['rma_control_head_hidden_dims']
        attn_feature_output = kwargs['attn_feature_output']
        critic_hidden_dims = kwargs['critic_hidden_dims']
        critic_attn_hidden_dims = kwargs['critic_attn_hidden_dims']
        critic_attn_output = kwargs['critic_attn_output']
        attn_output = kwargs['attn_output']

        activation = resolve_nn_activation(activation)
        mlp_input_dim_a = num_actor_obs 
        mlp_input_dim_c = num_critic_obs

        # Policy
        class Actor(nn.Module):
            def __init__(
                    self, 
                    mlp_input_dim_a, 
                    obs_feature, 
                    attn_feature, 
                    activation, 
                    actor_attn_hidden_dims, 
                    num_actions,
                    rma_actor_hidden_dims,
                    rma_control_head_hidden_dims,
                    attn_output):
                
                super().__init__()
               
                self.actor_rma = Actor_RMA(
                    mlp_input_dim_a=mlp_input_dim_a,
                    actor_hidden_dims=rma_actor_hidden_dims,
                    activation=activation,
                    control_head_hidden_dims=rma_control_head_hidden_dims,
                    num_actions=num_actions,
                    obs_feature=obs_feature,
                    attn_feature=attn_feature,
                    actor_attn_hidden_dims = actor_attn_hidden_dims,
                    attn_feature_output = attn_feature_output,
                    attn_output=attn_output
                )

            def forward(self, obs, obs_attn):
                output_rma = self.actor_rma(obs, obs_attn)
                output = output_rma # + output_attn
                return output
            
        # Value function
        class Critic(nn.Module):
            def __init__(self, mlp_input_dim_c,critic_obs_feature, critic_attn_feature, critic_attn_hidden_dims, critic_attn_output, critic_hidden_dims, activation ):
                super().__init__()

                self.critic_attn = Attention(
                    obs_feature=critic_obs_feature,
                    attn_feature=critic_attn_feature,
                    hidden_dims=critic_attn_hidden_dims,
                    out_dim=attn_feature_output,  
                    output_dim=critic_attn_output,                
                    activation=activation,
                    )    

                self.critic_control_head = CriticMLP(
                    input_dim=mlp_input_dim_c + critic_attn_output,
                    hidden_dims=critic_hidden_dims,  
                    output_dim=1,                
                    activation=activation,)    

            def forward(self, critic_obs, critic_attn):
                latent, _ = self.critic_attn(critic_obs, critic_attn)
                input = torch.cat([critic_obs, latent], dim=1)
                output = self.critic_control_head(input)
                return output

        self.actor  = Actor(mlp_input_dim_a, 
                            obs_feature, 
                            attn_feature, 
                            activation, 
                            actor_attn_hidden_dims, 
                            num_actions, 
                            rma_actor_hidden_dims, 
                            rma_control_head_hidden_dims,
                            attn_output)

        self.critic = Critic(mlp_input_dim_c, 
                             critic_obs_feature,
                             critic_attn_feature,
                             critic_attn_hidden_dims,
                             critic_attn_output,
                             critic_hidden_dims, 
                             activation)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter((init_noise_std * torch.ones(num_actions)).unsqueeze(0)) #TODO

        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)
        
    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs, obs_attn):
        # compute mean
        mean = self.actor(obs, obs_attn)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        std = torch.clamp(std, min=self.min_std)
        self.distribution = Normal(mean, std)

    def act(self, obs, obs_attn, **kwargs):
        self.update_distribution(obs, obs_attn)
        return self.distribution.sample()

    def act_inference(self, obs,obs_attn):
        actions_mean = self.actor(obs, obs_attn=obs_attn)
        return actions_mean

    def evaluate(self, critic_obs, critic_attn_obs, **kwargs):
        # obs = self.get_critic_obs(obs)
        return self.critic(critic_obs, critic_attn_obs)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                        module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True  # training resumes