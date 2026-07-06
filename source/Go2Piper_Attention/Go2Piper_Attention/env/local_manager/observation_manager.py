from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.managers import ObservationManager as ObservationManagerBase


class ObservationManager(ObservationManagerBase):

    def __init__(self,cfg, env):
        super().__init__(cfg, env)
        # self.num_leg_obs: int = 0  # The number of proprioceptive observations.

        # self.num_arm_obs: int = 0   # The number of privileged observations.

        self.num_history: int = 0  # The length of history.

        # self.num_critic_obs: int = 0

    # TODO:
    def compute_obs(self):
        # print("ObservationManager compute_obs:")
        for group_name in self._group_obs_term_names:
            # print("group_name",group_name)
            # check ig group name is valid
            if group_name not in self._group_obs_term_names:
                raise ValueError(
                    f"Unable to find the group '{group_name}' in the observation manager."
                    f" Available groups are: {list(self._group_obs_term_names.keys())}"
                )
            # iterate over all the terms in each group
            group_term_names = self._group_obs_term_names[group_name]
            # print("group_term_names",group_term_names)
            # read attributes for each term
            obs_terms = zip(group_term_names, self._group_obs_term_cfgs[group_name])  
            # print("obs_terms",obs_terms)
            for term_name, term_cfg in obs_terms:
                # print("term_name", term_name)
                # print("term_cfg",term_cfg)
                # if term_name.startswith("priv_"):
                #     self.num_priv += (term_cfg.func(self._env, **term_cfg.params).clone()).shape[1]
                # else:
                #     self.num_prop += (term_cfg.func(self._env, **term_cfg.params).clone()).shape[1]
                #     self.num_history = term_cfg.history_length 
                self.num_history = term_cfg.history_length 
        # return self.num_history, self.num_leg_obs, self.num_arm_obs, self.num_critic_obs
        return self.num_history