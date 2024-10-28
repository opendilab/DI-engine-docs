==============================
加载预训练模型与断点续训
==============================

在使用 DI-engine 进行强化学习实验时，加载预训练的 ``ckpt`` 文件进行断点续训是常见需求。本文将以 ``cartpole_ppo_config.py`` 为例，详细说明如何在 DI-engine 中加载预训练模型，并实现断点续训。

加载预训练模型
================

配置 ``load_path``
------------------

加载预训练模型的步骤首先在配置文件中指定需要加载的 ``ckpt`` 文件路径。这个路径通过 ``policy.load_path`` 字段进行配置。

示例::

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
            # ==========  指定预训练模型的ckpt路径 ==========
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
        # 或者你可以使用命令行方式运行 `ding -m serial_onpolicy -c cartpole_ppo_config.py -s 0`
        from ding.entry import serial_pipeline_onpolicy
        serial_pipeline_onpolicy((main_config, create_config), seed=0)

在上面的例子中，``policy.load_path`` 明确指定了预训练模型的路径 ``/path/to/your/ckpt/iteration_100.pth.tar``。当你运行这段代码时，DI-engine 会自动加载该路径下的模型权重，并在此基础上继续训练。

模型加载流程
----------------

模型加载的具体流程发生在 ``serial_entry_onpolicy.py`` 文件中，相关代码如下::

    # Load pretrained model if specified
    if cfg.policy.load_path is not None:
        logging.info(f'Loading model from {cfg.policy.load_path} begin...')
        if cfg.policy.cuda and torch.cuda.is_available():
            policy.learn_mode.load_state_dict(torch.load(cfg.policy.load_path, map_location='cuda'))
        else:
            policy.learn_mode.load_state_dict(torch.load(cfg.policy.load_path, map_location='cpu'))
        logging.info(f'Loading model from {cfg.policy.load_path} end!')

当 ``cfg.policy.load_path`` 不为空时，DI-engine 会加载指定路径下的预训练模型。若 ``cfg.policy.cuda`` 为 ``True`` 且 CUDA 可用，模型将被加载到 GPU 上，否则加载到 CPU。

断点续训
========

续训日志与 TensorBoard 路径
----------------------------

默认情况下，当你加载模型并继续训练时，DI-engine 会为新的实验过程创建一个新的路径。这样可以避免与之前的训练日志和 TensorBoard 数据冲突。如果你希望将断点续训的日志和 TensorBoard 数据保存在原来的路径下，可以在 ``serial_entry_onpolicy.py`` 中通过 ``renew_dir=False`` 来指定。

相关代码如下::

    cfg = compile_config(cfg, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg, save_cfg=True, renew_dir=False)

此时，DI-engine 会将断点续训的日志与之前的日志保存在同一文件夹下。然而，这种方式并不推荐，原因如下：

1. **迭代计数问题**：断点续训后的 ``iter/steps`` 会从 0 开始重新计数，与之前保存的训练数据产生混淆。
2. **TensorBoard 数据冲突**：在同一个 TensorBoard 文件中显示之前的学习曲线与断点续训后的学习曲线，可能会导致曲线交叠，影响可视化效果。

因此，推荐使用默认的新建路径方式，将续训过程中的日志和 TensorBoard 数据保存在一个新的文件夹中。

总结
====

- **加载预训练模型**：通过在配置文件中设置 ``policy.load_path`` 来指定预训练的 ``ckpt`` 文件路径。DI-engine 会在训练开始时自动加载该模型。
- **断点续训的路径管理**：默认情况下，续训时会新建一个带有时间戳的文件夹来保存新的训练日志和 TensorBoard 数据。如果你希望保存在原来的文件夹下，可以设置 ``renew_dir=False``，但不推荐这样做，以避免训练数据混乱。