Specifications of DI-engine Config
==================================

To ensure the ease of use, readability, and extensibility of the config，config submitted by the developers should comply with the following specifications.

DI-engine 的 config 包括 main_config 和 create_config 两部分。

Example Link
------------

Examples of Deep Q-Network （DQN）：

https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/config/serial/pong/pong_dqn_config.py

包含模型或数据的算法 (SQIL) 的示例:

https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/config/serial/pong/pong_sqil_config.py

Details of the specification
--------

Specification of Grammar
~~~~~~~~

-  config should satisfy flake8 python syntax checking and yapf formatting.

Specification of naming 
~~~~~~~~

-  config.py 文件名，main_config 和 create_config 相关变量名

   -  统一以<环境名>\_<算法名>\_config.py
      命名。文件的名称以及文件中相关变量名不用添加 default 字段。例如应该将文件名 hopper_onppo_default_config.py 改为 hopper_onppo_config.py。

   -  类似
      ICM 算法这种，总的算法是论文提出的模块再结合某个基线算法，其对应的 config 名称，按照<环境名>\_<模块名>\_<基线算法名>\_config.py
      命名，例如 cartpole_icm_offppo_config.py

   -  算法如果有 on-policy 和 off-policy 的不同版本，统一在 config.py 文件名和文件中相关变量名，使用 onppo/offppo 区分 on-policy 和 off-policy 版的算法。例如对于 PPO 算法的 config,
      应该将 hopper_ppo_config.py 改成 hopper_onppo_config.py。

-  exp_name 字段

   -  main_config 中必须添加 exp_name 字段

   -  命名规范为环境+算法+seed，例如\ ``qbert_sqil_seed0``

-  Name of the file path

   -  参见 sqil 示例，并加上相应的注释。如果需要加载多个模型 model，则模型路径 (model_path) 变量按照以下方式命名：prefix1_model_path，prefix2_model_path，...,
      数据路径 (data_path) 变量命名类似。

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
   cartpole, pendulum, bitflip
   环境使用 base, 其他环境一般使用 subprocess：

   .. code:: python

      env_manager=dict(type='subprocess'),

-  保证 evaluator_env_num：n_evaluator_episode = 1:1 （ smac 环境例外）

-  在 main_config 的 env 字段中一般不应该包含 manager 字段
   (当不包含 manager 字段时，shared_memory 默认为 True)：

   -  smac 环境例外，由于状态维度问题，smac 需要设置 shared_memory=Fasle。

   -  smac 环境外的其他环境，如果由于状态维度问题运行报错，可以包含 manager 字段并设置 shared
      memory=False。

-  如果想开启/关闭 shared memory, 请在env.manager字段中进行控制

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

   -  Please apple secondary references to avoid circular
      import：use \ ``from ding.entry import serial_pipeline``\ instead of \ ``from ding.entry.serial_entry import serial_pipeline``

   -  Use\ ``[main_config, create_config]``
      to unify the style，If an algorithm needs to call other config，this convention can be waived。例如 imitation
      learning 算法需要引入专家 config，具体参见 sqil 的示例。

   -  每一个 config 必须有一个启动命令，且写成类似下面这种格式

      .. code:: python

         if ___name___ == "___main___":
             # or you can enter `ding -m serial -c cartpole_dqn_config.py -s 0`
             from ding.entry import serial_pipeline
             serial_pipeline([main_config, create_config], seed=0)

      -  注意\ ``from ding.entry import serial_pipeline``\ 这行不要写在文件开头，
         要写在\ ``if ___name___ == "___main___":``\ 下面。

   -  如果算法使用了不同的 serial_pipeline_X，
      需要在\ https://github.com/opendilab/DI-engine/blob/5d2beed4a8a07fb70599d910c6d53cf5157b133b/ding/entry/cli.py#L189\ 中添加相应的启动命令对应
      ``serial_X``\ 。

-  seed 在入口函数中设置，config 中不要包含 seed。

-  If the hyperparameters in the algorithm have a certain reasonable range, please write a comment on the corresponding hyperparameters of the algorithm config. For instance the alpha value of sqil：

   .. code:: python

      alpha=0.1,  # alpha: 0.08-0.12

-  Please make sure all parameters in config are valid ，unused keys should be deleted.

-  一般在 config 中不包含 TODO 项，如果确实有必要写进 config，需要写清楚内容，例如：TODO(name):
   xxx.
