你好，世界
============================

.. toctree::
   :maxdepth: 2

决策智能是人工智能领域最重要的方向，它的一般形式是使用一个智能体来处理来自一个环境的信息，并给出合理的反馈与响应，并让环境和智能体的状态向着设计者所期望的方向变化。

我们首先使用"月球着陆器"这个环境来介绍 DI-engine 中的智能体，以及它是如何与环境交互的。


让智能体运行起来
--------------

智能体本质上是一个接受输入，反馈输出的数学模型。它的模型由一个模型结构和一组模型参数构成。
在机器学习领域的实践中，我们会把模型写入存放在一个文件中，或是从一个文件中读出所需要的智能体模型。
这里我们提供了一个由 DI-engine 框架使用 DQN 算法训练的智能体模型：
`final.pth.tar <https://github.com/opendilab/DI-engine/blob/main/dizoo/classic_control/cartpole/config/cartpole_dqn_config.py>`_ \
只需要使用以下的代码，就可以让智能体动起来，记得要把函数中的模型地址换成本地保存的地址：

.. code-block:: python

    import gym
    import torch
    from easydict import EasyDict
    from ding.config import compile_config
    from ding.envs import DingEnvWrapper
    from ding.policy import DQNPolicy, single_env_forward_wrapper
    from ding.model import DQN
    from lunarlander_dqn_config import main_config, create_config


    def main(main_config: EasyDict, create_config: EasyDict, ckpt_path: str):
        main_config.exp_name = 'lunarlander_dqn_deploy'
        cfg = compile_config(main_config, create_cfg=create_config, auto=True)
        env = DingEnvWrapper(gym.make(cfg.env.env_id), EasyDict(env_wrapper='default'))
        env.enable_save_replay(replay_path='./lunarlander_dqn_deploy/video')
        model = DQN(**cfg.policy.model)
        state_dict = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict['model'])
        policy = DQNPolicy(cfg.policy, model=model).eval_mode
        forward_fn = single_env_forward_wrapper(policy.forward)
        obs = env.reset()
        returns = 0.
        while True:
            action = forward_fn(obs)
            obs, rew, done, info = env.step(action)
            returns += rew
            if done:
                break
        print(f'Deploy is finished, final epsiode return is: {returns}')

    if __name__ == "__main__":
        main(main_config=main_config, create_config=create_config, ckpt_path='./final.pth.tar')


从代码中可见，通过使用 torch.load 可以获得模型的 Pytorch 对象，然后使用 load_state_dict 即可将模型加载至 DI-engine 的 DQN 模型中。
然后将 DQN 模型加载到 DQN 策略中，使用评价模式的 forward_fn 函数，即可让智能体对环境状态 obs 产生反馈的动作 action 。
智能体的动作 action 会和环境进行一次交互，生成下一个时刻的环境状态 obs ，此次交互的奖励 rew ，环境是否结束的信号 done 以及其他信息 info 。
所有时刻的奖励值会被累加，作为本次智能体在这个任务中的总分。

.. note::
    你可以在日志中看到此次部署智能体的总分，以及可以在文件目录中看到此次视频的回放。

更好地评估智能体
------------------------

在强化学习中，智能体的成绩可能会随不同的初始状态而发生波动。因此，我们需要开设多个环境，从而多进行几次评估测试，来更好地为它打分。
DI-engine 设计了环境管理器 env_manager 来做到这一点，我们可以使用以下稍微更复杂一些的代码来做到这一点：

.. code-block:: python

    import os
    import gym
    import torch
    from tensorboardX import SummaryWriter
    from easydict import EasyDict

    from ding.config import compile_config
    from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, AdvancedReplayBuffer
    from ding.envs import BaseEnvManager, DingEnvWrapper
    from ding.policy import DQNPolicy
    from ding.model import DQN
    from ding.utils import set_pkg_seed
    from ding.rl_utils import get_epsilon_greedy_fn
    from dizoo.box2d.lunarlander.config.lunarlander_dqn_config import main_config, create_config

    # Get DI-engine form env class
    def wrapped_cartpole_env():
        return DingEnvWrapper(
            gym.make(main_config['env']['env_id']),
            EasyDict(env_wrapper='default'),
        )


    def main(cfg, seed=0):
        cfg['exp_name'] = 'lunarlander_dqn_eval'
        cfg = compile_config(
            cfg,
            BaseEnvManager,
            DQNPolicy,
            BaseLearner,
            SampleSerialCollector,
            InteractionSerialEvaluator,
            AdvancedReplayBuffer,
            save_cfg=True
        )
        cfg.policy.load_path = 'lunarlander_dqn_seed0/ckpt/final.pth.tar'
        evaluator_env_num = cfg.env.evaluator_env_num
        evaluator_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(evaluator_env_num)], cfg=cfg.env.manager)

        # switch save replay interface
        # evaluator_env.enable_save_replay(cfg.env.replay_path)
        evaluator_env.enable_save_replay(replay_path='./lunarlander_dqn_eval/video')

        # Set random seed for all package and instance
        evaluator_env.seed(seed, dynamic_seed=False)
        set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

        # Set up RL Policy
        model = DQN(**cfg.policy.model)
        policy = DQNPolicy(cfg.policy, model=model)
        policy.eval_mode.load_state_dict(torch.load(cfg.policy.load_path, map_location='cpu'))

        # Evaluate
        tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
        evaluator = InteractionSerialEvaluator(
            cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
        )
        evaluator.eval()

    if __name__ == "__main__":
        main(main_config)

.. note::
    DI-engine 的环境管理器在对多个环境进行并行评估的时候，还会一并统计奖励的平均值，最大值和最小值，以及一些算法相关的其它指标。


让智能体变得更强
--------------

使用 DI-engine 运行以下的代码，来获得上述测试中的智能体模型。
试试自己生成一个智能体模型，或许它会更强：


.. code-block:: python

    import gym
    from ditk import logging
    from ding.model import DQN
    from ding.policy import DQNPolicy
    from ding.envs import DingEnvWrapper, BaseEnvManagerV2, SubprocessEnvManagerV2
    from ding.data import DequeBuffer
    from ding.config import compile_config
    from ding.framework import task, ding_init
    from ding.framework.context import OnlineRLContext
    from ding.framework.middleware import OffPolicyLearner, StepCollector, interaction_evaluator, data_pusher, \
        eps_greedy_handler, CkptSaver, online_logger, nstep_reward_enhancer
    from ding.utils import set_pkg_seed
    from dizoo.box2d.lunarlander.config.lunarlander_dqn_config import main_config, create_config

    def main():
        logging.getLogger().setLevel(logging.INFO)
        cfg = compile_config(main_config, create_cfg=create_config, auto=True)
        ding_init(cfg)
        with task.start(async_mode=False, ctx=OnlineRLContext()):
            collector_env = SubprocessEnvManagerV2(
                env_fn=[lambda: DingEnvWrapper(gym.make(cfg.env.env_id)) for _ in range(cfg.env.collector_env_num)],
                cfg=cfg.env.manager
            )
            evaluator_env = SubprocessEnvManagerV2(
                env_fn=[lambda: DingEnvWrapper(gym.make(cfg.env.env_id)) for _ in range(cfg.env.evaluator_env_num)],
                cfg=cfg.env.manager
            )

            set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

            model = DQN(**cfg.policy.model)
            buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
            policy = DQNPolicy(cfg.policy, model=model)

            task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
            task.use(eps_greedy_handler(cfg))
            task.use(StepCollector(cfg, policy.collect_mode, collector_env))
            task.use(nstep_reward_enhancer(cfg))
            task.use(data_pusher(cfg, buffer_))
            task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_))
            task.use(online_logger(train_show_freq=10))
            task.use(CkptSaver(cfg, policy, train_freq=100))
            task.run()

    if __name__ == "__main__":
        main()

至此您已经完成了 DI-engine 的 Hello World 任务，使用了提供的代码和模型，学习了强化学习的智能体与环境是如何交互的。
请继续阅读文档,来了解 DI-engine 的强化学习算法的生产框架是如何搭建的。
