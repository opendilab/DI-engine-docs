DI-engine Config Specification
===============================

The config submitted by the developer needs to comply with the following specifications to ensure the ease of use, readability, and extensibility of the config.

The config of DI-engine includes two parts: main_config and create_config.

Example Link
-------------

Example of a common algorithm (DQN):

https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/config/serial/pong/pong_dqn_config.py

Example of an algorithm (Soft Q Imitation Learning) containing a model or data:

https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/config/serial/pong/pong_sqil_config.py

Normative Content
------------------

Grammar Specification
~~~~~~~~~~~~~~~~~~~~~~

-  Config needs to satisfy flake8 python syntax check and yapf format.

Naming Conventions
~~~~~~~~~~~~~~~~~~~

-  config.py file name, main_config and create_config related variable names

   -  Unified with <environment name>\_<algorithm name>\_config.py. The name of the file and related variable names in the file do not need to add the default field. For example, the filename hopper_onppo_default_config.py should be changed to hopper_onppo_config.py.

   -  Similar For the ICM algorithm, the general algorithm is the module proposed in the paper combined with a baseline algorithm, and the corresponding config name is in accordance with <environment name>\_<module name>\_<baseline algorithm name>\_config.py Name it like cartpole_icm_offppo_config.py

   -  If the algorithm has different versions of on-policy and off-policy, unify the file name of config.py and the relevant variable name in the file, and use onppo/offppo to distinguish the algorithm of on-policy and off-policy version. For example, for the config of the PPO algorithm, hopper_ppo_config.py should be changed to hopper_onppo_config.py.

-  exp_name field

   -  exp_name field must be added in main_config

   -  The naming convention is environment+algorithm+seed, eg:\ ``qbert_sqil_seed0``

-  File pathname

   -  See the Soft Q Learning example and comment accordingly. If multiple models need to be loaded, the model path (model_path) variables are named as follows: prefix1_model_path, prefix2_model_path, ..., The data path (data_path) variable is named similarly.

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
~~~~~~~~~~~~~~~~~~~

-  For the env_manager field in create_config, except for simple environmentscartpole, pendulum, bitflip, The environment uses base, and other environments generally use subprocess:
 
   .. code:: python

      env_manager=dict(type='subprocess'),

-  Guaranteed evaluator_env_num: n_evaluator_episode = 1:1 (except for smac environment)

-  The manager field should generally not be included in the env field of main_config (shared_memory defaults to True when the manager field is not included):

   -  The smac environment is an exception. Due to the state dimension problem, smac needs to set shared_memory=Fasle.

   -  For other environments other than the smac environment, if an error is reported due to the state dimension problem, you can include the manager field and set shared memory=False.

-  If you want to enable/disable shared memory, please control it in the env.manager field

   .. code:: python

      config = dict(
          ...,
          env=dict(
              manager=dict(
                  shared_memory=True,
              ),
          ),
      )
-  Create Config

   -  In the env field, only need to include ``type`` and ``import_names``\ fields, E.g:

   .. code:: python

      env=dict(
          type='atari',
          import_names=['dizoo.atari.envs.atari_env'],
      ),

   -  The\ ``replay_buffer``\ field is generally not required. If you want to use the buffer stored as deque, please specify the type of replay_buffer as deque in create_config:

      .. code::

         replay_buffer=dict(type='deque'),

-  serial_pipeline

   -  Use secondary references to avoid circular import: i.e. use\ ``from ding.entry import serial_pipeline``\ instead of\ ``from ding.entry.serial_entry import serial_pipeline``

   -  Use \ ``[main_config, create_config]`` in a uniform style, if the algorithm needs to call other configs, it is not necessary to follow this convention. For example, the imitation learning algorithm needs to introduce expert config, see the example of Soft Q Learning for details.

   -  Each config must have a startup command written in a format similar to the following

      .. code:: python

         if ___name___ == "___main___":
             # or you can enter `ding -m serial -c cartpole_dqn_config.py -s 0` 
             from ding.entry import serial_pipeline
             serial_pipeline([main_config, create_config], seed=0)

   -  Note that \ ``from ding.entry import serial_pipeline``\ this line should not be written at the beginning of the file, but to be written below \ ``if ___name___ == "___main___":``\.

   -  If the algorithm uses a different serial_pipeline_X, Need to add the corresponding startup command in https://github.com/opendilab/DI-engine/blob/5d2beed4a8a07fb70599d910c6d53cf5157b133b/ding/entry/cli.py#L189\ ``serial_X``\ .

-  Seed is set in the entry function, do not include seed in config.

-  If the hyperparameters in the algorithm have a certain reasonable range, please write a comment on the corresponding hyperparameters in the algorithm config, such as the alpha value in SQIL:

   .. code:: python

      alpha=0.1,  # alpha: 0.08-0.12

-  Make sure all parameters in config are valid, you need to delete unused keys.

-  Generally, the TODO item is not included in the config. If it is really necessary to write it into the config, you need to write the content clearly, for example: TODO(name):

