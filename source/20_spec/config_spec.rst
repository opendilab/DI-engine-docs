Specifications of DI-engine Config
==================================

To ensure the ease of use, readability, and extensibility of the config，config submitted by the developers should comply with the following specifications.

Config of DI-engine includes two divisions：main_config and create_config.

Example Link
--------------

Example of Deep Q-Network （DQN）：

https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/config/serial/pong/pong_dqn_config.py

Example of SQIL with model or data:

https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/config/serial/pong/pong_sqil_config.py

Details of the specification
------------------------------

Specification of Grammar
~~~~~~~~~~~~~~~~~~~~~~~~~

-  config should satisfy flake8 python syntax checking and yapf formatting.

Specification of naming 
~~~~~~~~~~~~~~~~~~~~~~~~

-  file name： config.py，related variable name：main_config 和 create_config

   -  Uniformly named after <环境名>\_<算法名>\_config.py
      .The name of the file and related variable names in the file do not need to add the default field. For instance file name hopper_onppo_default_config.py should be changed into hopper_onppo_config.py。

   -  Similarly
      For ICM algorithm，the general algorithm is the module proposed in the paper combined with a baseline algorithm,，其对应的 config 名称，按照<环境名>\_<模块名>\_<基线算法名>\_config.py
      命名，例如 cartpole_icm_offppo_config.py

   -  算法如果有 on-policy 和 off-policy 的不同版本，统一在 config.py 文件名和文件中相关变量名，使用 onppo/offppo 区分 on-policy 和 off-policy 版的算法。例如对于 PPO 算法的 config,
      应该将 hopper_ppo_config.py 改成 hopper_onppo_config.py。

-  exp_name field

   -  main_config must include exp_name filed

   -  The naming convention is environemnt+algorithm+seed，such as \ ``qbert_sqil_seed0``

-  Name of the file path

   -  See the sqil example, commented accordingly.If multiple models need to be loaded, the model path (model_path) variable is named as follows：prefix1_model_path，prefix2_model_path，...,
      varibles of data_path are named in same way.

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

Main Specification
~~~~~~~~~~~~~~~~~~~~

-  对于 create_config 中的 env_manager 字段，除了简单环境
   cartpole, pendulum, bitflip
   环境使用 base, 其他环境一般使用 subprocess：

   .. code:: python

      env_manager=dict(type='subprocess'),

-  Ensure evaluator_env_num：n_evaluator_episode = 1:1 （ smac environment 例外）

-  在 main_config 的 env 字段中一般不应该包含 manager 字段
   (shared_memory defaults to True when the manager field is not included)：

   -  smac 环境例外，由于状态维度问题，smac 需要设置 shared_memory=Fasle。

   -  smac 环境外的其他环境，如果由于状态维度问题运行报错，可以包含 manager 字段并设置 shared
      memory=False。

-  If you want to turn on/off shared memory, please control it in env.manager filed

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

   -  in env field，only ``type`` 和 ``import_names``\ 两个字段,
      Such as：

   .. code:: python

      env=dict(
          type='atari',
          import_names=['dizoo.atari.envs.atari_env'],
      ),

   -  nnormally field \ ``replay_buffer``\ is unnecessary。If you want to use the buffer stored as deque，please specify the type of replay_buffer如果想使用存储为deque的buffer，请在create_config中指定replay_buffer的type为deque：

      .. code::

         replay_buffer=dict(type='deque'),

-  serial_pipeline

   -  Please apple secondary references to avoid circular
      import：use \ ``from ding.entry import serial_pipeline``\ instead of \ ``from ding.entry.serial_entry import serial_pipeline``

   -  Use\ ``[main_config, create_config]``
      to unify the style，If an algorithm needs to call other config，this convention can be waived。Such as imitation
      learning algorithm needs to introduce expert config, see the example of sqil for details。

   -  Every config must have a starting command，and it's format should as below

      .. code:: python

         if ___name___ == "___main___":
             # or you can enter `ding -m serial -c cartpole_dqn_config.py -s 0`
             from ding.entry import serial_pipeline
             serial_pipeline([main_config, create_config], seed=0)

      -  Remember \ ``from ding.entry import serial_pipeline``\ this line should not as the head of the file，
         please note it at \ ``if ___name___ == "___main___":``\ below.

   -  If the algorithm use different serial_pipeline_X，
      you need to add \ https://github.com/opendilab/DI-engine/blob/5d2beed4a8a07fb70599d910c6d53cf5157b133b/ding/entry/cli.py#L189\ 中添加相应的启动命令对应
      ``serial_X``\ 。

-  seed is set in the entry function, do not include seed in config.

-  If the hyperparameters in the algorithm have a certain reasonable range, please write a comment on the corresponding hyperparameters of the algorithm config. For instance the alpha value of sqil：

   .. code:: python

      alpha=0.1,  # alpha: 0.08-0.12

-  Please make sure all parameters in config are valid ，unused keys should be deleted.

-  Normally TODO is not include in config，, if it is really necessary to write into config，please mark the content clearly，such as：TODO(name):
   xxx.
