DI-engine config规范
====================

开发者提交的config需要遵守一下约定，以保证config的易用性，可读性，与可扩展性。

DI-engine的config包括main_config和create_config 两部分。

示例链接
--------

普通算法(DQN)的示例：

https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/config/serial/pong/pong_dqn_config.py

包含模型或数据的算法(SQIL)的示例:

https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/config/serial/pong/pong_sqil_config.py

规范内容
--------

语法规范
~~~~~~~~

-  config需要满足flake8 python语法检查, 以及进行yapf格式化。

命名规范
~~~~~~~~

-  config.py文件名，main_config和create_config 相关变量名

   -  统一以<环境名>\_<算法名>\_config.py
      命名。文件的名称以及文件中相关变量名不用添加default字段。例如应该将文件名hopper_onppo_default_config.py改为hopper_onppo_config.py。

   -  类似
      ICM算法这种，总的算法是论文提出的模块再结合某个基线算法，其对应的config名称，按照<环境名>\_<模块名>\_<基线算法名>\_config.py
      命名，例如cartpole_icm_offppo_config.py

   -  算法如果有on-policy和off-policy的不同版本，统一在config.py文件名和文件中相关变量名，使用onppo/offppo区分on-policy和off-policy版的算法。例如对于PPO算法的config,
      应该将hopper_ppo_config.py改成hopper_onppo_config.py。

-  exp_name 字段

   -  main_config中必须添加exp_name字段

   -  命名规范为环境+算法+seed，例如\ ``qbert_sqil_seed0``

-  文件路径名

   -  参见sqil示例，并加上相应的注释。如果需要加载多个模型model，则模型路径(model_path)变量按照以下方式命名：prefix1_model_path，prefix2_model_path，...,
      数据路径(data_path)变量命名类似。

.. code:: python

   config = dict(
       ...,
       # Users should add their own model path here. Model path should lead to a model.
       # Absolute path is recommended.
       # In DI-engine, it is ``exp_name/ckpt/ckpt_best.pth.tar``.
       model_path='model_path_placeholder',
       # Users should add their own data path here. Data path should lead to a file to store data or load the stored data.
       # Absolute path is recommended.
       # In DI-engine, it is usually located in ``exp_name`` directory
       data_path='data_path_placeholder',
   )

主要规范
~~~~~~~~

-  对于 create_config 中的 env_manager 字段，除了简单环境
   cartpole，pendulum，bitflip
   环境使用base，其他环境一般使用subprocess：

   .. code:: python

      env_manager=dict(type='subprocess'),

-  保证evaluator_env_num：n_evaluator_episode = 1:1 （smac环境例外）

-  在main_config的env字段中一般不应该包含manager字段
   (当不包含manager字段时，shared_memory默认为True)：

   -  smac环境例外，由于状态维度问题，smac需要设置shared_memory=Fasle。

   -  smac环境外的其他环境，如果由于状态维度问题运行报错，可以包含manager字段并设置shared
      memory=False。

-  如果想开启/关闭 shared memory，请在env.manager字段中进行控制

   .. code:: python

      config = dict(
          ...,
          env=dict(
              manager=dict(
                  shared_memory=True,
              ),
          ),
      )

-  create config

   -  env字段中，只需要包含 ``type`` 和 ``import_names``\ 两个字段,
      例如：

   .. code:: python

      env=dict(
          type='atari',
          import_names=['dizoo.atari.envs.atari_env'],
      ),

   -  一般不需要\ ``replay_buffer``\ 字段。如果想使用存储为deque的buffer，请在create_config中指定replay_buffer的type为deque：

      .. code::

         replay_buffer=dict(type='deque'),

-  serial_pipeline

   -  使用二级引用避免circular
      import：即使用\ ``from ding.entry import serial_pipeline``\ 而不是\ ``from ding.entry.serial_entry import serial_pipeline``

   -  使用\ ``[main_config, create_config]``
      以统一风格，如果算法需要调用其他config，可以不遵循此约定。例如imitation
      learning算法需要引入专家config，具体参见sqil的示例。

   -  每一个config必须有一个启动命令，且写成类似下面这种格式

      .. code:: python

         if ___name___ == "___main___":
             # or you can enter `ding -m serial -c cartpole_dqn_config.py -s 0`
             from ding.entry import serial_pipeline
             serial_pipeline([main_config, create_config], seed=0)

      -  注意\ ``from ding.entry import serial_pipeline``\ 这行不要写在文件开头，
         要写在\ ``if ___name___ == "___main___":``\ 下面。

   -  如果算法使用了不同的serial_pipeline_X，
      需要在\ https://github.com/opendilab/DI-engine/blob/5d2beed4a8a07fb70599d910c6d53cf5157b133b/ding/entry/cli.py#L189\ 中添加相应的启动命令对应
      ``serial_X``\ 。

-  seed在入口函数中设置，config中不要包含seed。

-  如果算法中超参数有确定的一个合理范围，请在算法config的对应超参数上写上注释，例如sqil中的alpha值：

   .. code:: python

      alpha=0.1,  # alpha: 0.08-0.12

-  确保config中所有参数都是有效的，需要删除没有用到的key。

-  一般在config中不包含TODO项，如果确实有必要写进config，需要写清楚内容，例如：TODO(name):
   xxx.
