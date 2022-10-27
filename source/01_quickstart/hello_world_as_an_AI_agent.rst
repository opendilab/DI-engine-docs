Hello World
======================================

.. toctree::
   :maxdepth: 2

Decision intelligence is the most important direction in the field of artificial intelligence. 
Its general form is to use an agent to process information from an environment, give reasonable feedback and responses, and make the state of the environment changing as designer's expectations.

We first use the "lunarlander" environment to introduce the agent in the DI-engine and how it interacts with the environment.


Let the Agent Run
------------------------------

An agent is essentially a mathematical model that accepts input and feeds back output. 
Its model consists of a model structure and a set of model parameters.
In the practice in the field of machine learning, we will write the model into a file for saving, or read the model from that file for deploying.
Here we provide an agent model trained by the DI-engine framework using the DQN algorithm:
`final.pth.tar <https://github.com/opendilab/DI-engine/blob/main/dizoo/classic_control/cartpole/config/cartpole_dqn_config.py>`_ \
Just use the following code to make the agent run, remember to replace the model address in the function with the locally saved model file path:

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

As can be seen from the code, the Pytorch object of the model can be obtained by using torch.load, and then the model can be loaded into the DQN model of DI-engine using load_state_dict.
Then load the DQN model into the DQN policy, and use the forward_fn function of the evaluation mode to make the agent generate feedback action for the environmental state, obs.
The action of the agent will interact with the environment once to generate the environment state, obs, at the next moment, the reward, rew, of this interaction, the signal, done, of whether the environment is over, and other information, info.

.. note::
    You can see the total score of the deployed agent in the log, and you can see the replay video in the experiment folder.


To Better Evaluate Agents
------------------------------

In reinforcement learning, the performance of the agent may fluctuate with different initial states. 
Therefore, we need to set up multiple environments and run several more evaluation tests to better score it.
DI-engine designed the environment manager env_manager to do this, we can do this with the following slightly more complex code:

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
    When evaluating multiple environments in parallel, the environment manager of DI-engine will also count the average, maximum and minimum rewards, as well as other indicators related to some algorithms.

Make agents stronger
--------------

Run the following code using DI-engine to get the agent model in the above test.
Try generating an agent model yourself, maybe it will be stronger:


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

So far, you have completed the Hello World task of DI-engine, used the provided code and model, and learned how the reinforcement learning agent interacts with the environment.
Please continue to read this document, `First Reinforcement Learning Program <../01_quickstart/first_rl_program.html>`_ , to understand how the RL pipeline is built in DI-engine.
