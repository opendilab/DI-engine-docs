BipedalWalker
~~~~~~~

概述
=======

前进有奖励，移动到最远端可以累计300+积分。 如果机器人摔倒，它会得到 -100。 应用电机扭矩会消耗少量点数，越多的最优代理将获得更好的分数。
状态包括船体角速度、角速度、水平速度、垂直速度、关节位置和关节角速度、腿与地面的接触以及 10 次激光雷达测距仪测量。 状态向量中没有坐标。

.. image:: ./images/bipedal_walker.gif
   :align: center

安装
====

安装方法
--------

安装gym和box2d两个库即可，可以通过pip一键安装或结合DI-engine安装

.. code:: shell

   # Method1: Install Directly
   pip install gym
   pip install box2d
   # Method2: Install with DI-engine requirements
   cd DI-engine
   pip install ".[common_env]"

验证安装
--------

安装完成后，可以通过在Python命令行中运行如下命令验证安装成功：

.. code:: python

   import gym
   env = gym.make('BipedalWalker-v3')
   obs = env.reset()
   print(obs.shape)  # (24,)

镜像
----

DI-engine准备好了配备有框架本身和Atari环境的镜像，可通过\ ``docker pull opendilab/ding:nightly``\ 获取，或访问\ `docker
hub <https://hub.docker.com/repository/docker/opendilab/ding>`__\ 获取更多镜像

.. _变换前的空间原始环境）:

变换前的空间（原始环境）
========================

.. _观察空间-1:

观察空间
--------

-  表示状态环境的数组，具体尺寸为\ ``(24)``\ ，数据类型为\ ``float32``

.. _动作空间-1:

动作空间
--------

-  游戏操作按键空间，一般是大小为4的连续动作空间（范围[1,-1]），数据类型为\ ``float32``\ ，需要传入python或者numpy数组（例如 ``np.array([0.1, -0.8, 0.33, 0.])``\ ）

-  代理四特征动作向量控制其四个腿关节的扭矩运动；每条腿有两个关节，腰部和膝盖
.. _奖励空间-1:

奖励空间
--------

-  前进有奖励，累计300+积分到远端。 如果机器人摔倒，它会得到 -100。 应用电机扭矩消耗少量积分，更优的代理将获得更好的分数，奖励是一个\ ``float``\ 数值，范围[-400, 300]

关键事实
========


其他
====


随机种子
--------

-  环境中有两部分随机种子需要设置，一是原始环境的随机种子，二是各种环境变换使用到的随机库的随机种子（例如\ ``random``\ ，\ ``np.random``\ ）

-  对于环境调用者，只需通过环境的\ ``seed``\ 方法进行设置这两个种子，无需关心具体实现细节

-  环境内部的具体实现：对于原始环境的种子，在调用环境的\ ``reset``\ 方法内部，具体的原始环境\ ``reset``\ 之前设置

-  环境内部的具体实现：对于随机库种子，则在环境的\ ``seed``\ 方法中直接设置该值


存储录像
--------

在环境创建之后，重置之前，调用\ ``enable_save_replay``\ 方法，指定游戏录像保存的路径。环境会在每个episode结束之后自动保存本局的录像文件。（默认调用\ ``gym.wrapper.Monitor``\ 实现，依赖\ ``ffmpeg``\ ），下面所示的代码将运行一个环境episode，并将这个episode的结果保存在形如\ ``./video/xxx.mp4``\ 这样的文件中：

.. code:: python

    from easydict import EasyDict
    from dizoo.box2d.bipedalwalker.envs import BipedalWalkerEnv
    import numpy as np

    env = BipedalWalkerEnv(EasyDict({'act_scale': True, 'rew_clip': True, 'replay_path': './video'}))
    obs = env.reset()

    while True:
       action = np.random.rand(24)
       timestep = env.step(action)
       if timestep.done:
           print('Episode is over, final eval reward is: {}'.format(timestep.info['final_eval_reward']))
           break

DI-zoo可运行代码示例
====================

完整的训练配置文件在 `github
link <https://github.com/opendilab/DI-engine/tree/main/dizoo/box2d/bipedalwalker/config>`__
内，对于具体的配置文件，例如\ ``bipedalwalker_td3_config.py``\ ，使用如下的demo即可运行：

    .. code:: python

    bipedalwalker_td3_config = dict(
        env=dict(
            collector_env_num=1,
            evaluator_env_num=5,
            # (bool) Scale output action into legal range.
            act_scale=True,
            n_evaluator_episode=5,
            stop_value=300,
            rew_clip=True,
            replay_path=None,
        ),
        policy=dict(
            cuda=True,
            priority=False,
            model=dict(
                obs_shape=24,
                action_shape=4,
                twin_critic=True,
                actor_head_hidden_size=400,
                critic_head_hidden_size=400,
                actor_head_type='regression',
            ),
            learn=dict(
                update_per_collect=4,
                discount_factor=0.99,
                batch_size=128,
                learning_rate_actor=0.001,
                learning_rate_critic=0.001,
                target_theta=0.005,
                ignore_done=False,
                actor_update_freq=2,
                noise=True,
                noise_sigma=0.2,
                noise_range=dict(
                    min=-0.5,
                    max=0.5,
                ),
            ),
            collect=dict(
                n_sample=256,
                noise_sigma=0.1,
                collector=dict(collect_print_freq=1000, ),
            ),
            eval=dict(evaluator=dict(eval_freq=100, ), ),
            other=dict(replay_buffer=dict(replay_buffer_size=50000, ), ),
        ),
    )
    bipedalwalker_td3_config = EasyDict(bipedalwalker_td3_config)
    main_config = bipedalwalker_td3_config

    bipedalwalker_td3_create_config = dict(
        env=dict(
            type='bipedalwalker',
            import_names=['dizoo.box2d.bipedalwalker.envs.bipedalwalker_env'],
        ),
        env_manager=dict(type='base'),
        policy=dict(type='td3'),
    )
    bipedalwalker_td3_create_config = EasyDict(bipedalwalker_td3_create_config)
    create_config = bipedalwalker_td3_create_config

   if __name__ == '__main__':
       from ding.entry import serial_pipeline
       serial_pipeline((main_config, create_config), seed=0)


基准算法性能
============

-  平均奖励等于300视为较好的Agent

    - BipedalWalker + TD3
    .. image:: images/bipedalwalker_td3.png
     :align: center
