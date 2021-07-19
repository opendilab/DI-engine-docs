策略概述
===================

策略模式
^^^^^^^^^^^^^^^

1. 3个模式
    
    * ``learn_mode`` : 一组旨在更新和优化策略的函数/方法.

        下面是一个关于如何在串行中应用策略``learn_mode``的演示:

        .. image:: images/serial_learner.png
            :scale: 60%

    * ``collect_mode`` : 一组旨在探索和利用平衡地收集训练数据的函数/方法。

        下面是一个关于如何在串行中应用策略``collect_mode``的演示:

        .. image:: images/serial_collector.png
            :scale: 60%

    * ``eval_mode`` : 一组负责公平策略评估的功能/方法。

        下面是一个关于如何在串行中应用策略``eval_mode'`的演示:

        .. image:: images/serial_evaluator.png
            :scale: 60%

2. 一些定制的模式（由用户定义）

    * ``command_mode`` : 一组用于不同模式间信息控制的功能/方法.

    * ``league_mode`` : 一组关于自我博弈游戏league训练有关的功能/方法。.

    * ``trick_mode`` : 一组用于超参数自适应调整的函数/方法。

策略接口
^^^^^^^^^^^^^^^^^^^^

1. 共同的接口:

    * ``default_config`` : 该策略的默认配置

    * ``__init__`` : 基本的和常见的初始化，例如模型、在线策略、设备；也可以根据参数``enable_field``启动学习、收集或评估模式.

    * ``_create_model`` : 如果没有传入模型，根据``default_model``模式创建一个模型.

    * ``_set_attribute`` : 为策略设置一个属性.

    * ``_get_attribute`` : 获取该策略的属性.

    * ``sync_gradients`` : 同步多个GPU的梯度.

    * ``default_model`` : 如果没有传入模型，``_create_model``将调用此方法，默认创建一个模型.

2. 学习模式接口:

    * ``_forward_learn`` : 学习模式的正向方法.

    * ``_reset_learn`` : 重置与学习模式有关的变量（如果有的话）. 不需要强制执行.

    * ``_monitor_vars_learn`` : 在学习者训练过程中被监控的变量。这些变量将被打印到文本和tensorboard记录器中.

    * ``_state_dict_learn`` : 返回模型的当前状态字典.

    * ``_load_state_dict_learn`` : 加载一个状态数据到模型中.

3. 收集模式接口:

    * ``_forward_collect`` : 收集模式的正向方法.

    * ``_reset_collect`` : 如果有的话, 重置收集模式相关的变量. 不需要强行实现这一点.

    * ``_process_transition`` : 将设想的时间步骤和策略输出处理成一个过渡.

    * ``_get_train_sample`` : 从一连串的过渡中获取可用于训练的样本.

    * ``_state_dict_collect`` : 返回模型的当前状态字典.

    * ``_load_state_dict_collect`` : 加载一个状态数据到模型中.

4. 评价模式接口:

    * ``_forward_eval`` : 评价模式的正向方法.

    * ``_reset_eval`` : 重置评估模式相关的变量（如果有的话）. 不需要强行实现这一点.

    * ``_state_dict_eval`` : 返回模型的当前状态字典.

    * ``_load_state_dict_eval`` : 加载一个状态数据到模型中.


以上提到的都是一些基本的定义和说明，用户可以从例子中学习.( ``ding/policy/`` )

.. 备注::
     **如何定义自己的get_train_sample案例？**

.. tip::
     **如何定义策略配置？**


    你可以参考 `this <../key_concept/index.html#config>`_. 这里我们以 ``default_config`` 的 ``DQNPolicy`` 为例.

    .. code:: python

        config = dict(
            # RL policy register name, refer to registry `POLICY_REGISTRY`.
            type='dqn',
            # Whether to use cuda for network
            cuda=False,
            # Whether the RL algorithm is on-policy or off-policy
            on_policy=False,
            # Whether use priority(Priority Experience Replay)
            priority=False,
            # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
            priority_IS_weight=False,
            # Reward’s future discount factor, aka. gamma
            discount_factor=0.97,
            # N-step reward discount sum for target q_value estimation
            nstep=1,
            # learn_mode policy config
            learn=dict(
                # (bool) Whether to use multi gpu
                multi_gpu=False,
                # How many updates(iterations) to train after collector's one collection.
                # Bigger "update_per_collect" means bigger off-policy.
                # collect data -> update policy-> collect data -> ...
                update_per_collect=3,
                # The number of samples of an iteration
                batch_size=64,
                # Gradient step length of an iteration.
                learning_rate=0.001,
                # ==============================================================
                # The following configs are algorithm-specific
                # ==============================================================
                # Frequence of target network update.
                target_update_freq=100,
                # Whether ignore done(usually for max step termination env)
                ignore_done=False,
                # Specific config for learner.
                learner=dict(),
            ),
            # collect_mode policy config
            collect=dict(
                # Only one of [n_sample, n_episode] shoule be set
                # n_sample=8,
                # Cut trajectories into pieces with length "unroll_len".
                unroll_len=1,
                # Specific config for collector.
                collector=dict(),
            ),
            # eval_mode policy config
            eval=dict(
                # Specific config for evaluator.
                evaluator=dict(),
            ),
            # other config
            other=dict(
                # Epsilon greedy with decay.
                eps=dict(
                    type='exp',
                    start=0.95,
                    end=0.1,
                    decay=10000,
                ),
                # Config for replay buffer.s
                replay_buffer=dict(
                    replay_buffer_size=10000,
                ),
            ),
        )

.. 备注::
    **如何在不同模式下定制模型？**

    在大多数情况下，学习、收集和评估模式使用一个相同的模型. 然而，他们可能会用不同的包装器来包装这个共享模型，以满足他们自己的需求. 比如说, 模型在收集和评估模式中不需要更新，而在学习模式中需要更新；收集模式模型可能需要使用探索，而评估模式模型不需要.

    然而，在一些策略中，不同模式的模型是不同的. 例如，逆强化学习需要一个专家模式来收集专家数据，然后用专家数据来训练一个新的模型.在这种情况下，用户需要在不同模式下定制模型.

    在正常的策略中，``_init_collect``模式中的模型启动可能是这样的。

    .. code:: python

        # `self.model` is initialized in policy base class's `__init__` mothod.
        self._collect_model = model_wrap(self.model, wrapper_name='base')
    
    而在策略``ILPolicy`中，`_init_collect`的方法是这样的。

    .. code:: python

        # FootballKaggle5thPlaceModel is an expert model.s
        self._collect_model = model_wrap(FootballKaggle5thPlaceModel(), wrapper_name='base')

.. 提示::
    许多算法使用目标模型(target model)来解决过度估计(over estimation)问题. 在策略中，也经常以这种方式实现.
    
    .. code:: python

        from ding.model import model_wrap
        
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='assign',
            # `policy.learn.target_update_freq`: Frequence of target network update. Int type.
            update_kwargs={'freq': self._cfg.learn.target_update_freq}
        )
