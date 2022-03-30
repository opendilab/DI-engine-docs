Partially Observable Markov Decision Process(POMDP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

概述
=======

介绍部分可观测马尔可夫决策过程，需要了解很多基础知识。

 -  马尔可夫性质，如果一个状态的下一个状态只取决于它当前状态，而跟它当前状态之前的状态都没有关系，即这个状态转移是符合马尔可夫的。
 -  马尔可夫过程，如果一个状态转移过程满足马尔可夫性质，即为马尔可夫过程。
 -  马尔可夫奖励过程，在马尔可夫过程的基础上，当没达到一个状态便获得一个奖励值，即为马尔可夫奖励过程。
 -  马尔可夫决策过程，相对于马尔可夫奖励过程添加了决策的步骤。

Atari是最经典最常用的离散动作空间强化学习环境，常作为离散空间强化学习算法的基准测试环境。
它是一个由Pong，Space，Invaders，QBert，Enduro，Breakout，MontezumaRevenge等57个子环境构成的集合。
在Atari环境中，v0后缀的环境表示以一定的概率重复之前的动作，不受Agent的控制。v4后缀表示这个概率为0。
同一环境的不同命名则表示Agent每隔多少帧才做一个动作，这个动作会在接下来的k帧中保持，以避免Agent能超出人类的反应上限。
一般而言，「Breakout-v0」表示跳过2到4帧，每一步随机选择跳过的帧数，「Deterministic」，如「BreakoutDeterministic-v0」表示恒定跳过4帧（除SpaceInvaders跳过3帧），「NoFrameskip」则表示不做跳过。

安装
====

安装方法
--------

安装gym和ale-py两个库即可，可以通过pip一键安装或结合DI-engine安装

注：atari-py库目前已被开发者废弃，建议使用\ `ale-py <https://github.com/mgbellemare/Arcade-Learning-Environment>`__

.. code:: python

   # Method1: Install Directly
   pip install gym
   pip install ale-py==0.7
   pip install autorom
   AutoROM --accept-license

验证安装
--------

运行如下Python命令，如果没有报错则说明安装成功。

.. code:: python

    import gym
    env = gym.make('Pong-ramNoFrameskip-v4')
    obs = env.reset()
    print(obs.shape)  # (128,)

镜像
----

DI-engine准备好了配备有框架本身和Atari环境的镜像，可通过\ ``docker pull opendilab/ding:nightly-atari``\ 获取，或访问\ `docker
hub <https://hub.docker.com/repository/docker/opendilab/ding>`__\ 获取更多镜像

.. _变换前的空间原始环境）:

变换前的空间（原始环境）
========================


观察空间
--------

-  由于是从ram中读取，所以会直接读取一维数据，数据具体尺寸为\(128)\ ，数据类型为\uint8\。


动作空间
--------

-  游戏操作按键空间，一般是大小为N的离散动作空间（N随具体子环境变化），数据类型为\ ``int``\ ，需要传入python数值（或是0维np数组，例如动作3为\ ``np.array(3)``\ ）

-  如在Pong环境中，N的大小为6，即动作在0-5中取值，具体的含义是：

   -  0：NOOP

   -  1：UP

   -  2：LEFT

   -  3：RIGHT

   -  4：DOWN

   -  5：FIRE

.. _奖励空间-1:

奖励空间
--------

-  游戏得分，根据具体游戏内容不同会有非常大的差异，一般是一个\ ``float``\ 数值，具体的数值可以参考最下方的基准算法性能部分。

.. _其他-1:

其他
----

-  游戏结束即为当前环境episode结束

关键事实
========

1. 2D
   虽然是RGB三通道图像输入，但需要堆叠多帧图像来解决单帧图像蕴含的信息不足（例如运动方向）。

2. 离散动作空间

3. 既有稠密奖励（Space
   Invaders）；又有稀疏奖励（Pitfall，MontezumaRevenge）

4. 奖励取值尺度变化较大，游戏奖励范围默认为 ``[-inf, inf]``。

.. _变换后的空间rl环境）:

变换后的空间（RL环境）
======================


观察空间
--------

-  变换内容：灰度图，空间尺寸缩放，最大最小值归一化，堆叠相邻N个游戏帧（N=4）

-  变换结果：三维np数组，尺寸为\ ``(4, 84, 84)``\ ，即为相邻的4帧灰度图，数据类型为\ ``np.float32``\ ，取值为 ``[0, 1]``

-  将MDP环境转换为POMDP过程主要是通过在观测信息obs中以一定概率增加noisy，以及在叠帧过程中以一定的概率复制前一时刻的图像帧。


动作空间
--------

-  基本无变换，依然是大小为N的离散动作空间，但一般为一维np数组，尺寸为\ ``(1, )``\ ，数据类型为\ ``np.int64``


奖励空间
--------

-  变换内容：奖励缩放和截断

-  变换结果：一维np数组，尺寸为\ ``(1, )``\ ，数据类型为\ ``np.float32``\ ，取值为 ``[-1, 1]``

上述空间使用gym环境空间定义则可表示为：

.. code:: python

   import gym
   obs_space = gym.spaces.Box(low=0, high=1, shape=(128,), dtype=np.float32)
   act_space = gym.spaces.Discrete(6)
   rew_space = gym.spaces.Box(low=-1, high=1, shape=(1, ), dtype=np.float32)


在Gym.spaces中，Box表示连续空间，
Discrete表示离散空间，
MultiBinary表示多维01空间，
MultiDiscrete表示多维离散空间，
Tuple表示Space元组，
Dict表示Space字典。其源码可查看
\ `Gym Spaces <https://github.com/openai/gym/tree/master/gym/spaces>`__。

其他
====

惰性初始化
----------

为了便于支持环境向量化等并行操作，环境实例一般实现惰性初始化，即\ ``__init__``\ 方法不初始化真正的原始环境实例，只是设置相关参数和配置值，在第一次调用\ ``reset``\ 方法时初始化具体的原始环境实例。

随机种子
--------

-  环境中有两部分随机种子需要设置，一是原始环境的随机种子，二是各种环境变换使用到的随机库的随机种子（例如\ ``random``\ ，\ ``np.random``\ ）

-  对于环境调用者，只需通过环境的\ ``seed``\ 方法进行设置这两个种子，无需关心具体实现细节

-  环境内部的具体实现：对于原始环境的种子，在调用环境的\ ``reset``\ 方法内部，具体的原始环境\ ``reset``\ 之前设置

-  环境内部的具体实现：对于随机库种子，则在环境的\ ``seed``\ 方法中直接设置该值

训练和测试环境的区别
--------------------

-  训练环境使用动态随机种子，即每个episode的随机种子都不同，都是由一个随机数发生器产生，但这个随机数发生器的种子是通过环境的\ ``seed``\ 方法固定的；测试环境使用静态随机种子，即每个episode的随机种子相同，通过\ ``seed``\ 方法指定。

-  训练环境和测试环境使用的环境预处理wrapper不同，\ ``episode_life``\ 和\ ``clip_reward``\ 在测试时不使用。

存储录像
--------

在环境创建之后，重置之前，调用\ ``enable_save_replay``\ 方法，指定游戏录像保存的路径。环境会在每个episode结束之后自动保存本局的录像文件。（默认调用\ ``gym.wrapper.Monitor``\ 实现，依赖\ ``ffmpeg``\ ），下面所示的代码将运行一个环境episode，并将这个episode的结果保存在形如\ ``./video/xxx.mp4``\ 这样的文件中：

.. code:: python

   from easydict import EasyDict
   from dizoo.atari.envs import AtariEnv

   env = AtariEnv(EasyDict({'env_id': 'Pong-ramNoFrameskip-v4', 'is_train': False}))
   env.enable_save_replay(replay_path='./video')
   obs = env.reset()

   while True:
       action = env.random_action()
       timestep = env.step(action)
       if timestep.done:
           print('Episode is over, final eval reward is: {}'.format(timestep.info['final_eval_reward']))
           break

DI-zoo可运行代码示例
====================

完整的训练配置文件在 `github
link <https://github.com/opendilab/DI-engine/tree/main/dizoo/pomdp/entry/>`__
内，对于具体的配置文件，例如\ ``pomdp_dqn_default_config.py``\ ，使用如下的demo即可运行：

.. code:: python

    from ding.entry import serial_pipeline
    from easydict import EasyDict

    pong_dqn_config = dict(
        env=dict(
            collector_env_num=8,
            evaluator_env_num=8,
            n_evaluator_episode=8,
            stop_value=20,
            env_id='Pong-ramNoFrameskip-v4',
            frame_stack=4,
            warp_frame=False,
            use_ram=True,
            pomdp=dict(noise_scale=0.01, zero_p=0.2, reward_noise=0.01, duplicate_p=0.2),
            manager=dict(shared_memory=False, )
        ),
        policy=dict(
            cuda=True,
            priority=False,
            model=dict(
                obs_shape=[
                    512,
                ],
                action_shape=6,
                encoder_hidden_size_list=[128, 128, 512],
            ),
            nstep=3,
            discount_factor=0.99,
            learn=dict(
                update_per_collect=10,
                batch_size=32,
                learning_rate=0.0001,
                target_update_freq=500,
            ),
            collect=dict(n_sample=100, ),
            eval=dict(evaluator=dict(eval_freq=4000, )),
            other=dict(
                eps=dict(
                    type='exp',
                    start=1.,
                    end=0.05,
                    decay=250000,
                ),
                replay_buffer=dict(replay_buffer_size=100000, ),
            ),
        ),
    )
    pong_dqn_config = EasyDict(pong_dqn_config)
    main_config = pong_dqn_config
    pong_dqn_create_config = dict(
        env=dict(
            type='pomdp',
            import_names=['di_zoo.pomdp.envs.atari_env'],
        ),
        env_manager=dict(type='subprocess'),
        policy=dict(type='dqn'),
    )
    pong_dqn_create_config = EasyDict(pong_dqn_create_config)
    create_config = pong_dqn_create_config

    if __name__ == '__main__':
        serial_pipeline((main_config, create_config), seed=0)

注：对于某些特殊的算法，比如PPO，需要使用专门的入口函数，示例可以参考
`link <https://github.com/opendilab/DI-engine/blob/main/dizoo/pomdp/entry/pomdp_ppo_default_config.py>`__

