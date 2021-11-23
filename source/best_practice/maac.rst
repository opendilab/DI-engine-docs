Multi-Agent Actor-Critic RL
============================================
MARL algorithms can be divided into two broad categories: centralized learning and decentralized learning. Recent work has developed two lines of research to bridge the gap betewwn these two frameworks: centralized training and decentralized execution(CTDE) and value decomposition(VD).
VD such as Qmix typically represents the joint Q-function as a function of agents’ local Q-functions, which has been considered as the gold standard for many MARL benchmarks.
CTDE methods such as MADDPG, MAPPO and COMA improve upon decentralized RL by adopting an actor-critic structure and learning a centralized critic. 
In DI-engine, we introduced the multi-agent actor-critic framework to quickly convert a single-agent algorithm into a multi-agent algorithm.


Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Unlike single-agent environments that return a Tensor-type observation, our multi-agent environments will return a dict-type observation, which includes agent_state, global_state and action_mask.

.. code:: 

   return {
         'agent_state': self.get_obs(),
         'global_state': self.get_global_special_state(),
         'action_mask': self.get_avail_actions(),
   }

- agent state: Agent state is ecah agent's local observation.
- global state: Global state contains all global information and the necessary agent-specific features, such as agent id, available actions.

Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Unlike single-agent environments that feed the same observation information to actor and critic networks, in multi-agent environments, we feed agent_state and action_mask information to the actor network to get each actions' logits and mask the invalid/inaccessible actions. 
At the same time, we feed global_state information to the critic network to gei the global critic value.

.. code:: 

    def compute_actor(self, x: torch.Tensor) -> Dict:
        action_mask = x['action_mask']
        x = x['agent_state']
        x = self.actor_encoder(x)
        x = self.actor_head(x)
        logit = x['logit']
        # action mask
        logit[action_mask == 0.0] = -99999999
        return {'logit': logit}

    def compute_critic(self, x: Dict) -> Dict:
        x = self.critic_encoder(x['global_state'])
        x = self.critic_head(x)
        return {'value': x['pred']}

Policy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When modifying the single-agent algorithm into a multi-agent algorithm, the policy part basically remains the same, the only thing to note is to add the multi_agent key and to call the multi-agent model when the multi_agent key is True.

.. code:: 

    MAPPO:

    def default_model(self) -> Tuple[str, List[str]]:
        if self._cfg.multi_agent:
            return 'mappo', ['ding.model.template.mappo']
        else:
            return 'vac', ['ding.model.template.vac']

    MASAC:

    def default_model(self) -> Tuple[str, List[str]]:
        if self._cfg.multi_agent:
            return 'maqac', ['ding.model.template.maqac']
        else:
            return 'qac', ['ding.model.template.qac']

- Action Mask: In multi-agent games, it is often the case that some actions cannot be executed due to game constraints. For example, in SMAC, an agent may have skills that cannot be performed frequently. So, when computing the logits for the softmax action probability, we mask out the unavailable actions in both the forward and backward pass so that the probabilities for unavailable actions are always zero. We find that this substantially accelerates training.
- Death Mask: In multi-agent games, an agent may die before the game terminates, such as SAMC environment. Note that we can always access the game state to compute the agent-specific global state for those dead agents. Therefore, even if an agent dies and becomes inactive in the middle of a rollout, value learning can still be performed in the following timesteps using inputs containing information of other live agents. This is typical in many existing multi-agent PG implementations. Our suggestion is to simply use a zero vector with the agent’s ID as the input to the value function after an agent dies. We call this approach “Death Masking”.

Config
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Open the multi-agent key and just change the environment to the one you want to run. 

.. code:: 

   agent_num = 5
   collector_env_num = 8
   evaluator_env_num = 8
   special_global_state = True,

   main_config = dict(
      exp_name='smac_5m6m_ppo',
      env=dict(
         map_name='5m_vs_6m',
         difficulty=7,
         reward_only_positive=True,
         mirror_opponent=False,
         agent_num=agent_num,
         collector_env_num=collector_env_num,
         evaluator_env_num=evaluator_env_num,
         n_evaluator_episode=16,
         stop_value=0.99,
         death_mask=True,
         special_global_state=special_global_state,
         manager=dict(
               shared_memory=False,
               reset_timeout=6000,
         ),
      ),
      policy=dict(
         cuda=True,
         multi_agent=True,
         continuous=False,
         model=dict(
               # (int) agent_num: The number of the agent.
               # For SMAC 3s5z, agent_num=8; for 2c_vs_64zg, agent_num=2.
               agent_num=agent_num,
               # (int) obs_shape: The shapeension of observation of each agent.
               # For 3s5z, obs_shape=150; for 2c_vs_64zg, agent_num=404.
               # (int) global_obs_shape: The shapeension of global observation.
               # For 3s5z, obs_shape=216; for 2c_vs_64zg, agent_num=342.
               agent_obs_shape=72,
               #global_obs_shape=216,
               global_obs_shape=152,
               # (int) action_shape: The number of action which each agent can take.
               # action_shape= the number of common action (6) + the number of enemies.
               # For 3s5z, obs_shape=14 (6+8); for 2c_vs_64zg, agent_num=70 (6+64).
               action_shape=12,
               # (List[int]) The size of hidden layer
               # hidden_size_list=[64],
         ),
         # used in state_num of hidden_state
         learn=dict(
               # (bool) Whether to use multi gpu
               multi_gpu=False,
               epoch_per_collect=10,
               batch_size=3200,
               learning_rate=5e-4,
               # ==============================================================
               # The following configs is algorithm-specific
               # ==============================================================
               # (float) The loss weight of value network, policy network weight is set to 1
               value_weight=0.5,
               # (float) The loss weight of entropy regularization, policy network weight is set to 1
               entropy_weight=0.01,
               # (float) PPO clip ratio, defaults to 0.2
               clip_ratio=0.05,
               # (bool) Whether to use advantage norm in a whole training batch
               adv_norm=False,
               value_norm=True,
               ppo_param_init=True,
               grad_clip_type='clip_norm',
               grad_clip_value=10,
               ignore_done=False,
         ),
         on_policy=True,
         collect=dict(env_num=collector_env_num, n_sample=3200),
         eval=dict(env_num=evaluator_env_num),
      ),
   )
   main_config = EasyDict(main_config)
   create_config = dict(
      env=dict(
         type='smac',
         import_names=['dizoo.smac.envs.smac_env'],
      ),
      env_manager=dict(type='base'),
      policy=dict(type='ppo'),
   )
   create_config = EasyDict(create_config)


The following are the parameters for each map of the SMAC environment.

+------------------+---------------------+---------------------+---------------------+
| Map              | agent_obs_shape     | global_obs_shape    | action_shape        |
+==================+=====================+=====================+=====================+
| 3s5z             | 150                 | 295                 | 14                  |
+------------------+---------------------+---------------------+---------------------+
| 5m_vs_6m         | 72                  | 152                 | 12                  |
+------------------+---------------------+---------------------+---------------------+
| MMM              | 186                 | 389                 | 16                  |
+------------------+---------------------+---------------------+---------------------+
| MMM2             | 204                 | 431                 | 18                  |
+------------------+---------------------+---------------------+---------------------+
| 2c_vs_64zg       | 404                 | 671                 | 70                  |
+------------------+---------------------+---------------------+---------------------+
| 6h_vs_8z         | 98                  | 209                 | 14                  |
+------------------+---------------------+---------------------+---------------------+
| 3s5z_vs_3s6z     | 159                 | 314                 | 15                  |
+------------------+---------------------+---------------------+---------------------+
| 27m_vs_30m       | 348                 | 1454                | 36                  |
+------------------+---------------------+---------------------+---------------------+