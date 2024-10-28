==============================
Loading Pretrained Models and Resuming Training
==============================

In DI-engine, it is common to load a pretrained ``ckpt`` file and resume training from a checkpoint. This document will take ``cartpole_ppo_config.py`` as an example to explain how to load a pretrained model and implement resume training in DI-engine.

Loading Pretrained Models
==========================

Configuring ``load_path``
-------------------------

To load a pretrained model, you need to specify the path to the ``ckpt`` file in the configuration file. This can be done by setting the ``policy.load_path`` field.

Example::

    from easydict import EasyDict

    cartpole_ppo_config = dict(
        exp_name='cartpole_ppo_seed0_loadckpt',
        env=dict(
            collector_env_num=8,
            evaluator_env_num=5,
            n_evaluator_episode=5,
            stop_value=195,
        ),
        policy=dict(
            # ========== Path to the pretrained ckpt ==========
            load_path='/path/to/your/ckpt/iteration_100.pth.tar',
            cuda=False,
            action_space='discrete',
            model=dict(
                obs_shape=4,
                action_shape=2,
                action_space='discrete',
                encoder_hidden_size_list=[64, 64, 128],
                critic_head_hidden_size=128,
                actor_head_hidden_size=128,
            ),
            learn=dict(
                epoch_per_collect=2,
                batch_size=64,
                learning_rate=0.001,
                value_weight=0.5,
                entropy_weight=0.01,
                clip_ratio=0.2,
                learner=dict(hook=dict(save_ckpt_after_iter=100)),
            ),
            collect=dict(
                n_sample=256,
                unroll_len=1,
                discount_factor=0.9,
                gae_lambda=0.95,
            ),
            eval=dict(evaluator=dict(eval_freq=100, ), ),
        ),
    )
    cartpole_ppo_config = EasyDict(cartpole_ppo_config)
    main_config = cartpole_ppo_config
    cartpole_ppo_create_config = dict(
        env=dict(
            type='cartpole',
            import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
        ),
        env_manager=dict(type='base'),
        policy=dict(type='ppo'),
    )
    cartpole_ppo_create_config = EasyDict(cartpole_ppo_create_config)
    create_config = cartpole_ppo_create_config

    if __name__ == "__main__":
        # Or you can run it via command line `ding -m serial_onpolicy -c cartpole_ppo_config.py -s 0`
        from ding.entry import serial_pipeline_onpolicy
        serial_pipeline_onpolicy((main_config, create_config), seed=0)

In the example above, ``policy.load_path`` specifies the pretrained model path as ``/path/to/your/ckpt/iteration_100.pth.tar``. When you run this code, DI-engine will automatically load the model weights and continue training from that point.

Model Loading Process
----------------------

The model loading process occurs in the ``serial_entry_onpolicy.py`` file. The related code is as follows::

    # Load pretrained model if specified
    if cfg.policy.load_path is not None:
        logging.info(f'Loading model from {cfg.policy.load_path} begin...')
        if cfg.policy.cuda and torch.cuda.is_available():
            policy.learn_mode.load_state_dict(torch.load(cfg.policy.load_path, map_location='cuda'))
        else:
            policy.learn_mode.load_state_dict(torch.load(cfg.policy.load_path, map_location='cpu'))
        logging.info(f'Loading model from {cfg.policy.load_path} end!')

When ``cfg.policy.load_path`` is not None, DI-engine will load the pretrained model from the specified path. If ``cfg.policy.cuda`` is ``True`` and CUDA is available, the model will be loaded onto the GPU; otherwise, it will be loaded onto the CPU.

Resuming Training
==================

Logging and TensorBoard Path for Resumed Training
--------------------------------------------------

By default, when you load a model and resume training, DI-engine will create a new path for the new training process. This avoids conflicts with previous training logs and TensorBoard data. However, if you want the resumed training logs and TensorBoard data to be saved in the original path, you can set ``renew_dir=False`` in ``serial_entry_onpolicy.py``.

The relevant code is as follows::

    cfg = compile_config(cfg, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg, save_cfg=True, renew_dir=False)

This will save the resumed logs in the same folder as the previous training. However, this is **not recommended**, for the following reasons:

1. **Iteration Counting**: After resuming, the ``iter/steps`` will start counting from 0, which may confuse the previous training data.
2. **TensorBoard Data Confusion**: Displaying both the previous learning curve and the new curve after resuming training in the same TensorBoard file may result in overlapping curves, making the visualization unclear.

Therefore, it is recommended to keep the default behavior and create a new directory for the logs and TensorBoard files of the resumed training.

Summary
=======

- **Loading Pretrained Models**: You can specify the pretrained ``ckpt`` file path in the configuration by setting ``policy.load_path``. DI-engine will automatically load the model at the start of training.
- **Managing Paths for Resumed Training**: By default, a new directory with a timestamp will be created for logs and TensorBoard files during resumed training. If you want to save logs in the original directory, you can set ``renew_dir=False``, though this is not recommended to avoid confusion with the training data.