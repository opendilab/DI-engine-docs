dmc2gym
~~~~~~~~~~~~

概述
=======

dmc2gym 是针对\ `DeepMind Control Suite <https://github.com/deepmind/dm_control>`__\ 的轻量级wrapper，提供标准的 OpenAI Gym 接口。
DeepMind Control Suite 是一组具有标准化结构和可解释奖励的连续控制任务，旨在作为强化学习代理的性能基准。


.. image:: ./images/dmc2gym.png
   :align: center

安装
====

安装方法
--------

需安装 gym 、 dm_control 和 dmc2gym , 用户可以选择通过 pip 一键安装（这里我不确定）

注：如果用户没有 root 权限，请在 install 的命令后面加上 ``--user``


.. code:: shell

   # Install Directly
   pip install gym
   pip install dm_control
   pip install git+git://github.com/denisyarats/dmc2gym.git

验证安装
--------

安装完成后，可以通过在 Python 命令行中运行如下命令验证安装成功：

.. code:: python

   import dmc2gym
    env = dmc2gym.make(domain_name='point_mass', task_name='easy', seed=1)
    obs = env.reset()
    print(obs.shape)    # (4,)

镜像
----

DI-engine 的镜像配备有框架本身和 dmc2gym 环境，可通过\ ``docker pull opendilab/ding:nightly-dmc2gym``\ 获取，或访问\ `docker hub <https://hub.docker.com/r/opendilab/ding>`__\ 获取更多镜像

环境介绍
========================

选择任务
----------------

dm_control 包含多个任务，我们这里暂时实现了如下任务：

-  Ball in cup (8, 2, 8)
   
   平面球杯任务。一个被驱动的平面容器可以在垂直平面上平移，以便摆动并接住一个连接在其底部的球。当球在杯子里时，接球任务的奖励为 1，否则为 0。

   -  catch
  
-  Cart-pole (4, 1, 5)

   通过在其底部向推车施加力来摆动并平衡未驱动的杆。本环境实现了如下任务

   -  balance: 初始杆靠近立柱

   -  swingup: 初始杆指向下方

-  Cheetah (18, 6, 17)
  
   平面的奔跑中的两足动物，奖励\ ``r``\ 与前向速度 \ ``v``\ 成线性比例，最大为 10m/s，即 \ ``r(v) = max(0, min(v/10, 1))``\

   -  run

-  Finger (6, 2, 12)
   
   基于 xxxpaper 的 3 自由度玩具操纵问题。 平面上用一个“手指”在无其他驱动力的铰链上旋转物体，使得自由体的尖端必须与目标重叠。

   -  spin: 在此任务中，物体必须不断地旋转。

-  Reacher (4, 2, 7)

   具有随机目标位置的简单两连杆平面伸展器。 奖励是杆末端执行器穿透目标球体时的奖励。

   -  easy: 目标球体比在困难任务中更大

-  Walker (18, 6, 24)

   基于 xxxpaper 中介绍的改进的平面步行器。 walk 任务包括一个鼓励前进速度的组件。

   -  walk

通过设置参数\ ``domain_name``\ ,\ ``task_name``\进行调用：

-  例如

.. code:: python

    env = DMC2GymEnv(EasyDict({
        "domain_name": "cartpole",
        "task_name": "balance",
    }))


-  按照论文中的任务，相应的状态空间、动作空间、观察空间\ ``(dim(S), dim(A), dim(O))``\ 如下表所示：

+------------+----------+------------+------------+-----------+
|   Domain   |   Task   |   dim(S)   |   dim(A)   |   dim(O)  |
+============+==========+============+============+===========+
|ball in cup |catch     |8           |2           |8          |
+------------+----------+------------+------------+-----------+
|cart-pole   |balance   |4           |1           |5          |
+            +----------+------------+------------+-----------+
|            |swingup   |4           |1           |5          |
+------------+----------+------------+------------+-----------+
|cheetah     |run       |18          |6           |7          |
+------------+----------+------------+------------+-----------+
|finger      |spin      |6           |2           |12         |
+------------+----------+------------+------------+-----------+
|reacher     |easy      |4           |2           |7          |
+------------+----------+------------+------------+-----------+
|walker      |walk      |18          |6           |24         |
+------------+----------+------------+------------+-----------+



观察空间
----------------

基于图像观察 
^^^^^^^^^^^^^^^^^^^^^^^^^

-  即当设置\ ``from_pixels=True``\时，观察空间为三通道，长宽分别为height, width的游戏图像

-  可以通过设置cfg中的\ ``height, width``\ 参数调整所观察图像尺寸。

-  通过设置\ ``channels_first``\ 来决定观察空间的具体shape

   -  \ ``channels_first=True``\观察空间shape为[3, height, width]

   -  \ ``channels_first=False``\ ，观察空间shape为[3, height, width]

-  每个channel的单个像素值范围为\ ``[0, 255]``\ ， 数据类型为\ ``uint8``\

非基于图像观察 
^^^^^^^^^^^^^^^^^^^^^^^^^

-  即当设置\ ``from_pixels=False``\时，观察空间维度遵循上述表格的中\ ``dim(O)``\ 

-  默认范围为 \ ``[-inf, inf]``\ 

动作空间
--------

-  动作空间维度遵循上述表格的中\ ``dim(A)``\

-  dmc2gym 对动作空间进行了标准化，每个维度动作空间的范围是\ ``[-1, 1]``\ ，类型为\ ``float32``\ 。

奖励空间
--------

基于图像观察 
^^^^^^^^^^^^^^^^^^^^^^^^^

-  与 cfg 设置的\ ``frame_skip``\ 参数有关，即表示每一步基于\ ``frame_skip``\ 帧的图像，维度为\ ``1``\

-  范围为\ ``[0, frame_skip]``\ ，类型为\ ``float32``\ ，默认\ ``frame_skip = 1``\

   -  即每帧画面的奖励空间为 [0, 1] ，\ ``frame_skip``\ 对奖励进行了进行叠加（这个说法不是很好）

非基于图像观察 
^^^^^^^^^^^^^^^^^^^^^^^^^

-  维度为\ ``1``\ ，范围 [0, 1] ，类型为\ ``float32``\

其他
====

游戏结束
---------------------

什么时候算结束？

随机种子
----------------------

暂时没有很懂，但是有dynamic_seed

存储录像
----------------------

在环境创建之后，重置之前，调用\ ``enable_save_replay``\ 方法，指定游戏录像保存的路径。环境会在每个episode结束之后自动保存本局的录像文件。（默认调用\ ``gym.wrapper.Monitor``\ 实现，依赖\ ``ffmpeg``\ ），下面所示的代码将运行一个环境episode，并将这个episode的结果保存在形如\ ``./video/xxx.mp4``\ 这样的文件中：

.. code:: python

   from easydict import EasyDict
   from dizoo.dmc2gym.envs import DMC2GymEnv

   env = DMC2GymEnv(EasyDict({
        "domain_name": "cartpole",
        "task_name": "balance",
        "frame_skip": 2,
        "from_pixels": True,
    }))
   env.enable_save_replay(replay_path='./video')
   env.seed(314, dynamic_seed=False)
   obs = env.reset()

   while True:
       action = env.random_action()
       timestep = env.step(action)
       if timestep.done:
           print('Episode is over, final eval reward is: {}'.format(timestep.info['final_eval_reward']))
           break


DI-zoo 可运行代码示例
======================

! 这里的config有点问题

完整的示例文件在 `github
link <https://github.com/opendilab/DI-engine/blob/main/dizoo/dmc2gym/entry/dmc2gym_save_replay_example.py>`__
内

.. code:: python

    from easydict import EasyDict

    cartpole_balance_ddpg_config = dict(
        exp_name='dmc2gym_cartpole_balance_ddpg_eval',
        env=dict(
            env_id='dmc2gym_cartpole_balance',
            domain_name='cartpole',
            task_name='balance',
            from_pixels=False,
            norm_obs=dict(use_norm=False, ),
            norm_reward=dict(use_norm=False, ),
            collector_env_num=1,
            evaluator_env_num=8,
            use_act_scale=True,
            n_evaluator_episode=8,
            replay_path='./dmc2gym_cartpole_balance_ddpg_eval/video',
            stop_value=1000,
        ),
        policy=dict(
            cuda=True,
            random_collect_size=2560,
            load_path="./dmc2gym_cartpole_balance_ddpg/ckpt/iteration_10000.pth.tar",
            model=dict(
                obs_shape=5,
                action_shape=1,
                twin_critic=False,
                actor_head_hidden_size=128,
                critic_head_hidden_size=128,
                action_space='regression',
            ),
            learn=dict(
                update_per_collect=1,
                batch_size=128,
                learning_rate_actor=1e-3,
                learning_rate_critic=1e-3,
                ignore_done=False,
                target_theta=0.005,
                discount_factor=0.99,
                actor_update_freq=1,
                noise=False,
            ),
            collect=dict(
                n_sample=1,
                unroll_len=1,
                noise_sigma=0.1,
            ),
            other=dict(replay_buffer=dict(replay_buffer_size=10000, ), ),
        )
    )
    cartpole_balance_ddpg_config = EasyDict(cartpole_balance_ddpg_config)
    main_config = cartpole_balance_ddpg_config

    cartpole_balance_create_config = dict(
        env=dict(
            type='dmc2gym',
            import_names=['dizoo.dmc2gym.envs.dmc2gym_env'],
        ),
        env_manager=dict(type='base'),
        policy=dict(
            type='ddpg',
            import_names=['ding.policy.ddpg'],
        ),
        replay_buffer=dict(type='naive', ),
    )
    cartpole_balance_create_config = EasyDict(cartpole_balance_create_config)
    create_config = cartpole_balance_create_config


基准算法性能
==============

-  dmc2gym

   - Cartpole Balance + DDPG

-  等结果

文档问题
==============

! 需要说一下可以cfg调整的参数吗

! dim(S)需要吗？

! 开头的简介好像需要再多点？感觉都没说清动作、奖励的含义等等，简单说一下？

! 各任务基于paper需要写出来吗？图可以用吗