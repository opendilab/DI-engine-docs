如何构建多智能体环境
==============================================================

``DI-zoo`` 为用户提供了一些多智能体强化学习常用环境，包括 `PettingZoo Env <https://github.com/opendilab/DI-engine/blob/main/dizoo/petting_zoo/envs/petting_zoo_simple_spread_env.py>`_ 、 `SMAC Env <https://github.com/opendilab/DI-engine/blob/main/dizoo/smac/envs/smac_env.py>`_ 、 `Multi-Agent MuJoCo Env <https://github.com/opendilab/DI-engine/blob/main/dizoo/multiagent_mujoco/envs/multi_mujoco_env.py>`_ 、 `Google Research Football Env <https://github.com/opendilab/DI-engine/blob/main/dizoo/gfootball/envs/gfootball_academy_env.py>`_ 。

然而在部分情况下，用户需要依据自己的业务，自己实现多智能体环境，并期待可以将其快速迁移到 ``DI-engine`` 中，以便使用 ``DI-engine`` 中的多智能体强化学习算法解决问题。本节将会介绍构建自己的多智能体强化学习环境，并利用``DI-engine``中的多智能体算法（以 `QMIX <https://github.com/opendilab/DI-engine/blob/main/ding/policy/qmix.py>`_ 与 `MAPPO <https://github.com/opendilab/DI-engine/blob/main/ding/policy/ppo.py>`_ 这两类用于合作多智能体的CTDE算法为例）来训练多智能体的方式。下面大部分情况下将以 PettingZoo 为例进行说明。具体代码可以参考 `PettingZoo Env <https://github.com/opendilab/DI-engine/blob/main/dizoo/petting_zoo/envs/petting_zoo_simple_spread_env.py>`_ 。

多智能体环境的构建方法和单智能体环境的构建方式基本是一样的，因此首先需要对``DI-engine`` 中强化学习环境的 `构建方式 <https://di-engine-docs.readthedocs.io/zh_CN/latest/04_best_practice/ding_env_zh.html>`_ 进行了解，新建的环境文件需要在 ``dizoo`` 文件夹下，也需要实现 ``__init__()`` 、 ``seed()`` 、 ``reset()`` 与 ``step()`` 等方法。

特殊的是，相较于单智能体环境，多智能体环境的动作空间、奖励空间和观测空间是字典，用以区分不同智能体的元素，拿``PettingZoo``的动作空间和奖励空间举例：


.. code:: python

    self._action_space = gym.spaces.Dict({agent: self._env.action_space(agent) for agent in self._agents})
    ...
    self._reward_space = gym.spaces.Dict(
        {
            agent: gym.spaces.Box(low=float("-inf"), high=float("inf"), shape=(1, ), dtype=np.float32)
            for agent in self._agents
        }
    )

具体而言，在 ``reset()`` 方法中，利用 ``gym.spaces.Dict`` 类定义动作空间与奖励空间，包含每个智能体的动作和奖励子空间。

此外，多智能体环境的观测空间， ``observation_space`` 则往往更加复杂，在CTDE的算法框架下，往往至少要包含两个部分，即 ``agent_state`` 与 ``global_state`` ，其中：

    - ``agent_state`` 代表每个智能体的 **局部** 观测，用于在执行过程中进行决策；
    - ``global_state`` 代表 **全局** 状态，用于在训练中缓解多智能体的非平稳问题。往往是智能体观测

因此在多智能体环境中，特别需要额外关注就是观测空间 ``observation space``，不同智能体的观测、全局观测以及其它自定义的观测形式。以 PettingZoo 环境为例，其 ``reset()``函数中，这样定义观测空间：


.. code:: python

    self._observation_space = gym.spaces.Dict(
        {
        'agent_state': gym.spaces.Box(
            low=float("-inf"),
            high=float("inf"),
            shape=(self._num_agents,
                    self._env.observation_space('agent_0').shape[0]),  # (self._num_agents, 30)
            dtype=np.float32
        ),
        'global_state': gym.spaces.Box(
            low=float("-inf"),
            high=float("inf"),
            shape=(
                4 * self._num_agents + 2 * self._num_landmarks + 2 * self._num_agents *
                (self._num_agents - 1),
            ),
            dtype=np.float32
        ),
        'agent_alone_state': gym.spaces.Box(
            low=float("-inf"),
            high=float("inf"),
            shape=(self._num_agents, 4 + 2 * self._num_landmarks + 2 * (self._num_agents - 1)),
            dtype=np.float32
        ),
        'agent_alone_padding_state': gym.spaces.Box(
            low=float("-inf"),
            high=float("inf"),
            shape=(self._num_agents,
                    self._env.observation_space('agent_0').shape[0]),  # (self._num_agents, 30)
            dtype=np.float32
        ),
        'action_mask': gym.spaces.Box(
            low=float("-inf"),
            high=float("inf"),
            shape=(self._num_agents, self._action_dim[0]),  # (self._num_agents, 5)
            dtype=np.float32
        )
        }
    )

即在每次环境返回的 observation 都需要返回一个字典，其中包含``agent_state``、``global_state``等信息。这些信息最终在模型前传的过程中被使用。


.. code:: python

    def _process_obs(self, obs: 'torch.Tensor') -> np.ndarray:  # noqa
        obs = np.array([obs[agent] for agent in self._agents]).astype(np.float32)
        if self._cfg.get('agent_obs_only', False):
            return obs
        ret = {}
        # Raw agent observation structure is --
        # [self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, communication]
        # where `communication` are signals from other agents (two for each agent in `simple_spread_v2`` env)

        # agent_state: Shape (n_agent, 2 + 2 + n_landmark * 2 + (n_agent - 1) * 2 + (n_agent - 1) * 2).
        #              Stacked observation. Contains
        #              - agent itself's state(velocity + position)
        #              - position of items that the agent can observe(e.g. other agents, landmarks)
        #              - communication
        ret['agent_state'] = obs
        # global_state: Shape (n_agent * (2 + 2) + n_landmark * 2 + n_agent * (n_agent - 1) * 2, ).
        #               1-dim vector. Contains
        #               - all agents' state(velocity + position) +
        #               - all landmarks' position +
        #               - all agents' communication
        ret['global_state'] = np.concatenate(
            [
                obs[0, 2:-(self._num_agents - 1) * 2],  # all agents' position + all landmarks' position
                obs[:, 0:2].flatten(),  # all agents' velocity
                obs[:, -(self._num_agents - 1) * 2:].flatten()  # all agents' communication
            ]
        )
        # agent_specific_global_state: Shape (n_agent, 2 + 2 + n_landmark * 2 + (n_agent - 1) * 2 + (n_agent - 1) * 2 + n_agent * (2 + 2) + n_landmark * 2 + n_agent * (n_agent - 1) * 2).
        #               2-dim vector. contains
        #               - agent_state info
        #               - global_state info
        if self._agent_specific_global_state:
            ret['global_state'] = np.concatenate(
                [ret['agent_state'],
                    np.expand_dims(ret['global_state'], axis=0).repeat(self._num_agents, axis=0)],
                axis=1
            )
        # agent_alone_state: Shape (n_agent, 2 + 2 + n_landmark * 2 + (n_agent - 1) * 2).
        #                    Stacked observation. Exclude other agents' positions from agent_state. Contains
        #                    - agent itself's state(velocity + position) +
        #                    - landmarks' positions (do not include other agents' positions)
        #                    - communication
        ret['agent_alone_state'] = np.concatenate(
            [
                obs[:, 0:(4 + self._num_agents * 2)],  # agent itself's state + landmarks' position
                obs[:, -(self._num_agents - 1) * 2:],  # communication
            ],
            1
        )
        # agent_alone_padding_state: Shape (n_agent, 2 + 2 + n_landmark * 2 + (n_agent - 1) * 2 + (n_agent - 1) * 2).
        #                            Contains the same information as agent_alone_state;
        #                            But 0-padding other agents' positions.
        ret['agent_alone_padding_state'] = np.concatenate(
            [
                obs[:, 0:(4 + self._num_agents * 2)],  # agent itself's state + landmarks' position
                np.zeros((self._num_agents,
                            (self._num_agents - 1) * 2), np.float32),  # Other agents' position(0-padding)
                obs[:, -(self._num_agents - 1) * 2:]  # communication
            ],
            1
        )
        # action_mask: All actions are of use(either 1 for discrete or 5 for continuous). Thus all 1.
        ret['action_mask'] = np.ones((self._num_agents, *self._action_dim))
        return ret

因此，在 ``reset()`` 与 ``step()`` 函数中，当获取到 observation 时，需要将 observation 处理为符合 observation_space 的内容后才能返回。对应于 ``PettingZoo`` 环境中的 ``_process_obs()`` 函数。同理， ``action`` 与 ``reward`` 也要经过处理后才能传入环境或返回给智能体。


.. code:: python

    action = self._process_action(action)
    ...
    rew_n = np.array([sum([rew[agent] for agent in self._agents])])
    ...
    return BaseEnvTimestep(obs_n, rew_n, done_n, info)

如何使用 ``DI-engine`` 中的 MARL 算法
``DI-engine`` 中集成了多种多智能体算法，包括 value-based 的 `QMIX <https://github.com/opendilab/DI-engine/blob/main/ding/policy/qmix.py>`_ 、 `QTRAN <https://github.com/opendilab/DI-engine/blob/main/ding/policy/qtran.py>`_ 以及actor-critic的 `COMA <https://github.com/opendilab/DI-engine/blob/main/ding/policy/coma.py>`_ 、 `MAPPO <https://github.com/opendilab/DI-engine/blob/main/ding/policy/ppo.py>`_ 等，这里以 QMIX 与 MAPPO 为例。

当环境已经完成后，进行智能体训练只需要修改默认算法配置文件的几个参数。以 ``PettingZoo`` 下的 QMIX config文件为例：


.. code:: python

    from easydict import EasyDict

    n_agent = 3
    n_landmark = n_agent
    collector_env_num = 8
    evaluator_env_num = 8
    main_config = dict(
        exp_name='ptz_simple_spread_qmix_seed0',
        env=dict(
            env_family='mpe',
            env_id='simple_spread_v2',
            n_agent=n_agent,
            n_landmark=n_landmark,
            max_cycles=25,
            agent_obs_only=False,
            continuous_actions=False,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            stop_value=0,
        ),
        policy=dict(
            cuda=True,
            model=dict(
                agent_num=n_agent,
                obs_shape=2 + 2 + n_landmark * 2 + (n_agent - 1) * 2 + (n_agent - 1) * 2,
                global_obs_shape=n_agent * 4 + n_landmark * 2 + n_agent * (n_agent - 1) * 2,
                action_shape=5,
                hidden_size_list=[128, 128, 64],
                mixer=True,
            ),
            learn=dict(
                update_per_collect=100,
                batch_size=32,
                learning_rate=0.0005,
                target_update_theta=0.001,
                discount_factor=0.99,
                double_q=True,
            ),
            collect=dict(
                n_sample=600,
                unroll_len=16,
                env_num=collector_env_num,
            ),
            eval=dict(env_num=evaluator_env_num, ),
            other=dict(eps=dict(
                type='exp',
                start=1.0,
                end=0.05,
                decay=100000,
            ), ),
        ),
    )
    main_config = EasyDict(main_config)
    create_config = dict(
        env=dict(
            import_names=['dizoo.petting_zoo.envs.petting_zoo_simple_spread_env'],
            type='petting_zoo',
        ),
        env_manager=dict(type='subprocess'),
        policy=dict(type='qmix'),
    )
    create_config = EasyDict(create_config)

    ptz_simple_spread_qmix_config = main_config
    ptz_simple_spread_qmix_create_config = create_config

    if __name__ == '__main__':
        # or you can enter `ding -m serial -c ptz_simple_spread_qmix_config.py -s 0`
        from ding.entry import serial_pipeline
        serial_pipeline((main_config, create_config), seed=0)

需要修改的内容有以下几点：
- main_config 的 env 属性：其中包含需要传递给实现的多智能体环境类的 ``__init__`` 函数的参数，包括子环境的的名称、智能体数量等；
- main_config 中 policy 的 model 属性：其中包含需要传递给模型的参数，包括智能体的局部观测维度、全局观测维度、动作维度等；
- create_config 的 env 属性，包含实现的多智能体环境所在的路径以及其在装饰器中的 key (type)。
其它的内容与环境无关，直接照搬就可以运行。

如果想要利用 actor-critic 的 MAPPO 算法，则需要对环境作额外的改动，由于 critic 需要对每个智能体的价值做判断，而之前的全局信息不包含智能体的判别信息，即 critic 无从得知这是要对哪个智能体做出评价，因此无法计算正确的价值。为此，在环境中需要使用 ``agent_specific_global_state`` 来替代原来的 ``global_state``。还是用 ``PettingZoo`` 环境作为例子：


.. code:: python

    if self._agent_specific_global_state:
        agent_specifig_global_state = gym.spaces.Box(
            low=float("-inf"),
            high=float("inf"),
            shape=(
                self._num_agents, self._env.observation_space('agent_0').shape[0] + 4 * self._num_agents +
                2 * self._num_landmarks + 2 * self._num_agents * (self._num_agents - 1)
            ),
            dtype=np.float32
        )
        self._observation_space['global_state'] = agent_specifig_global_state

所谓 ``agent_specific_global_state``，就是将智能体自己的局部观测与全局状态进行叠加，这样 ``global_state`` 就既有智能体的判别信息，也具有足够的全局信息来让 critic 给出正确的价值。
同理，在 ``reset()`` 与 ``step()`` 中处理 observation 时，也要修改返回的 ``global_state``：


.. code:: python

    if self._agent_specific_global_state:
        ret['global_state'] = np.concatenate(
            [ret['agent_state'],
                np.expand_dims(ret['global_state'], axis=0).repeat(self._num_agents, axis=0)],
            axis=1
        )

在环境修改完成后，同样对 config 文件做小的修改即可运行，以 PettingZoo 环境的 MAPPO 的配置文件为例：


.. code:: python

    from easydict import EasyDict

    n_agent = 3
    n_landmark = n_agent
    collector_env_num = 8
    evaluator_env_num = 8
    main_config = dict(
        exp_name='ptz_simple_spread_mappo_seed0',
        env=dict(
            env_family='mpe',
            env_id='simple_spread_v2',
            n_agent=n_agent,
            n_landmark=n_landmark,
            max_cycles=25,
            agent_obs_only=False,
            agent_specific_global_state=True,
            continuous_actions=False,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            stop_value=0,
        ),
        policy=dict(
            cuda=True,
            multi_agent=True,
            action_space='discrete',
            model=dict(
                action_space='discrete',
                agent_num=n_agent,
                agent_obs_shape=2 + 2 + n_landmark * 2 + (n_agent - 1) * 2 + (n_agent - 1) * 2,
                global_obs_shape=2 + 2 + n_landmark * 2 + (n_agent - 1) * 2 + (n_agent - 1) * 2 + n_agent * (2 + 2) +
                n_landmark * 2 + n_agent * (n_agent - 1) * 2,
                action_shape=5,
            ),
            learn=dict(
                multi_gpu=False,
                epoch_per_collect=5,
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
                clip_ratio=0.2,
                # (bool) Whether to use advantage norm in a whole training batch
                adv_norm=False,
                value_norm=True,
                ppo_param_init=True,
                grad_clip_type='clip_norm',
                grad_clip_value=10,
                ignore_done=False,
            ),
            collect=dict(
                n_sample=3200,
                unroll_len=1,
                env_num=collector_env_num,
            ),
            eval=dict(
                env_num=evaluator_env_num,
                evaluator=dict(eval_freq=50, ),
            ),
            other=dict(),
        ),
    )
    main_config = EasyDict(main_config)
    create_config = dict(
        env=dict(
            import_names=['dizoo.petting_zoo.envs.petting_zoo_simple_spread_env'],
            type='petting_zoo',
        ),
        env_manager=dict(type='subprocess'),
        policy=dict(type='ppo'),
    )
    create_config = EasyDict(create_config)
    ptz_simple_spread_mappo_config = main_config
    ptz_simple_spread_mappo_create_config = create_config

    if __name__ == '__main__':
        # or you can enter `ding -m serial_onpolicy -c ptz_simple_spread_mappo_config.py -s 0`
        from ding.entry import serial_pipeline_onpolicy
        serial_pipeline_onpolicy((main_config, create_config), seed=0)

相较于 QMIX 的改动外，唯一的区别就是增加了对于 ``agent_specific_global_state=True`` 的判断。
