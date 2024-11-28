在 DI-engine 中使用 DDP 分布式训练
==================================

当需要加速强化学习训练过程时，使用分布式数据并行 (Distributed Data Parallel, DDP) 是一个非常有效的方式。
本文将详细说明如何在 DI-engine 中配置和使用 DDP 训练，并通过 ``pong_dqn_ddp_config.py`` 示例进行讲解。

启动 DDP 训练
-------------

使用 PyTorch 的 `distributed.launch` 模块来启动 DDP 训练。运行以下命令::

    python -m torch.distributed.launch --nproc_per_node=2 --master_port=29501 ./dizoo/atari/config/serial/pong/pong_dqn_ddp_config.py

其中：

- ``--nproc_per_node=2``：指定使用 2 个 GPU 进行训练。
- ``--master_port=29501``：指定主进程的端口号。
- 最后的参数为配置文件的路径。

DDP 训练的配置说明
*************

相比单 GPU 训练 (如 ``pong_dqn_config.py``)，在启用 DDP 多 GPU 训练时，配置文件 ``pong_dqn_ddp_config.py`` 有以下两个重要的不同点：

1. **在 Policy 配置中启用多 GPU 支持**::

    policy=dict(
        multi_gpu=True,  # 启用多 GPU 训练模式
        cuda=True,       # 使用 CUDA 加速
        ...
    )

    - 多 GPU 训练模式的关键代码初始化位于 `base_policy.py` 中： 
      [base_policy.py#L167](https://github.com/opendilab/DI-engine/blob/main/ding/policy/base_policy.py#L167)
    - 在 ``policy._forward_learn()`` 中进行梯度同步：
      [dqn.py#L281](https://github.com/opendilab/DI-engine/blob/main/ding/policy/dqn.py#L281)

2. **使用 DDPContext 管理分布式训练流程**::

    if __name__ == '__main__':
        from ding.utils import DDPContext
        from ding.entry import serial_pipeline
        with DDPContext():
            serial_pipeline((main_config, create_config), seed=0, max_env_step=int(3e6))

    - ``DDPContext`` 用于初始化分布式训练环境，以及释放分布式训练资源。

DI-engine 中的 DDP 实现原理
***************************

DI-engine 的 DDP 实现主要包含以下几个关键部分：

1. **Collector 数据收集的分布式处理**::

    - 在 ``SampleSerialCollector`` 中，各进程独立收集数据样本。
    - 收集完成后，通过 ``allreduce`` 同步各进程的统计数据：
      [sample_serial_collector.py#L355](https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/sample_serial_collector.py#L355)::

        if self._world_size > 1:
            collected_sample = allreduce_data(collected_sample, 'sum')
            collected_step = allreduce_data(collected_step, 'sum')
            collected_episode = allreduce_data(collected_episode, 'sum')
            collected_duration = allreduce_data(collected_duration, 'sum')

2. **Evaluator 评估过程的分布式处理**::

    - 评估逻辑仅在 rank 0 进程上运行：
      [interaction_serial_evaluator.py#L207](https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/interaction_serial_evaluator.py#L207)::

        if get_rank() == 0:
            # 执行评估逻辑
            ...

    - 评估完成后，评估结果通过广播同步给其他进程：
      [interaction_serial_evaluator.py#L315](https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/interaction_serial_evaluator.py#L315)::

        if get_world_size() > 1:
            objects = [stop_flag, episode_info]
            broadcast_object_list(objects, src=0)
            stop_flag, episode_info = objects

3. **日志记录的分布式处理**::

    - 仅在 rank 0 进程上初始化日志记录器：
      [serial_entry.py#L72](https://github.com/opendilab/DI-engine/blob/main/ding/entry/serial_entry.py#L72)::

        tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial')) if get_rank() == 0 else None

    - 仅在 rank 0 进程上记录日志：
      [sample_serial_collector.py#L59](https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/sample_serial_collector.py#L59)::

        if self._rank == 0:
            if tb_logger is not None:
                self._logger, _ = build_logger(
                    path='./{}/log/{}'.format(self._exp_name, self._instance_name),
                    name=self._instance_name,
                    need_tb=False
                )
                self._tb_logger = tb_logger
            else:
                self._logger, self._tb_logger = build_logger(
                    path='./{}/log/{}'.format(self._exp_name, self._instance_name), name=self._instance_name
                )
        else:
            self._logger, _ = build_logger(
                path='./{}/log/{}'.format(self._exp_name, self._instance_name), name=self._instance_name, need_tb=False
            )
            self._tb_logger = None

    - 仅在 rank 0 进程上打印日志：
      [sample_serial_collector.py#L388](https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/sample_serial_collector.py#L388)::

        if self._rank != 0:
            return

总结
****

在 DI-engine 中，DDP 分布式训练通过分布式数据收集、评估和日志记录等模块，充分利用多 GPU 的计算能力以加速训练过程。
DDP 的核心逻辑依赖 PyTorch 分布式框架，同时通过 ``DDPContext`` 对分布式环境进行统一管理，简化了开发者在分布式训练中的配置和使用流程。

有关更多详细实现，可以参考以下代码链接：

- [base_policy.py#L167](https://github.com/opendilab/DI-engine/blob/main/ding/policy/base_policy.py#L167)
- [dqn.py#L281](https://github.com/opendilab/DI-engine/blob/main/ding/policy/dqn.py#L281)
- [sample_serial_collector.py#L355](https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/sample_serial_collector.py#L355)
- [interaction_serial_evaluator.py#L207](https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/interaction_serial_evaluator.py#L207)
- [interaction_serial_evaluator.py#L315](https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/interaction_serial_evaluator.py#L315)