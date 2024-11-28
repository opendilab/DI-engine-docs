# Distributed Training with DDP in DI-engine  
===========================================  

When accelerating the reinforcement learning (RL) training process, **Distributed Data Parallel (DDP)** is an effective approach.  
This article provides a detailed explanation of how to configure and use DDP training in **DI-engine**, demonstrated through the example `pong_dqn_ddp_config.py`.  

Launching DDP Training  
-----------------------  

Use PyTorch's `distributed.launch` module to start DDP training. Run the following command:  

```bash  
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29501 ./dizoo/atari/config/serial/pong/pong_dqn_ddp_config.py  
```  

Where:  

- `--nproc_per_node=2`: Specifies the use of 2 GPUs for training.  
- `--master_port=29501`: Sets the port number for the master process.  
- The final argument is the path to the configuration file.  

Key Configuration for DDP Training  
***********************************  

Compared to single-GPU training (e.g., `pong_dqn_config.py`), the DDP multi-GPU training configuration file `pong_dqn_ddp_config.py` has the following two key differences:  

1. **Enable multi-GPU support in the policy configuration**:  

    ```python  
    policy=dict(
        multi_gpu=True,  # Enable multi-GPU training mode  
        cuda=True,       # Use CUDA acceleration  
        ...
    )
    ```  

    - The key code for initializing multi-GPU training is located in `base_policy.py`:  
      [base_policy.py#L167](https://github.com/opendilab/DI-engine/blob/main/ding/policy/base_policy.py#L167)  
    - Gradient synchronization is performed in `policy._forward_learn()`:  
      [dqn.py#L281](https://github.com/opendilab/DI-engine/blob/main/ding/policy/dqn.py#L281)  

2. **Use `DDPContext` to manage the distributed training process**:  

    ```python  
    if __name__ == '__main__':
        from ding.utils import DDPContext
        from ding.entry import serial_pipeline
        with DDPContext():
            serial_pipeline((main_config, create_config), seed=0, max_env_step=int(3e6))
    ```  

    - `DDPContext` initializes the distributed training environment and releases resources when training ends.  

DDP Implementation in DI-engine  
*******************************  

DI-engine's DDP implementation includes the following key components:  

1. **Distributed Handling of Data Collection in the Collector**:  

    - In `SampleSerialCollector`, each process independently collects data samples.  
    - After collection, statistics are synchronized across all processes using `allreduce`:  
      [sample_serial_collector.py#L355](https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/sample_serial_collector.py#L355):  

        ```python  
        if self._world_size > 1:
            collected_sample = allreduce_data(collected_sample, 'sum')
            collected_step = allreduce_data(collected_step, 'sum')
            collected_episode = allreduce_data(collected_episode, 'sum')
            collected_duration = allreduce_data(collected_duration, 'sum')
        ```  

2. **Distributed Processing of Evaluation**:  

    - Evaluation logic only runs on the rank 0 process:  
      [interaction_serial_evaluator.py#L207](https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/interaction_serial_evaluator.py#L207):  

        ```python  
        if get_rank() == 0:
            # Perform evaluation logic  
            ...
        ```  

    - After evaluation, results are broadcasted to other processes:  
      [interaction_serial_evaluator.py#L315](https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/interaction_serial_evaluator.py#L315):  

        ```python  
        if get_world_size() > 1:
            objects = [stop_flag, episode_info]
            broadcast_object_list(objects, src=0)
            stop_flag, episode_info = objects
        ```  

3. **Distributed Handling of Logging**:  

    - The logger is only initialized on the rank 0 process:  
      [serial_entry.py#L72](https://github.com/opendilab/DI-engine/blob/main/ding/entry/serial_entry.py#L72):  

        ```python  
        tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial')) if get_rank() == 0 else None
        ```  

    - Logs are only recorded on the rank 0 process:  
      [sample_serial_collector.py#L59](https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/sample_serial_collector.py#L59):  

        ```python  
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
        ```  

    - Logs are only printed on the rank 0 process:  
      [sample_serial_collector.py#L388](https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/sample_serial_collector.py#L388):  

        ```python  
        if self._rank != 0:
            return
        ```  

Summary  
*******  

In DI-engine, DDP distributed training leverages distributed data collection, evaluation, and logging modules to fully utilize the computational power of multiple GPUs, accelerating the training process.  

The core DDP logic relies on PyTorch's distributed framework. Meanwhile, `DDPContext` simplifies the configuration and usage of distributed environments, providing a unified management interface for distributed training.  

For more implementation details, refer to the following code links:  

- [base_policy.py#L167](https://github.com/opendilab/DI-engine/blob/main/ding/policy/base_policy.py#L167)  
- [dqn.py#L281](https://github.com/opendilab/DI-engine/blob/main/ding/policy/dqn.py#L281)  
- [sample_serial_collector.py#L355](https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/sample_serial_collector.py#L355)  
- [interaction_serial_evaluator.py#L207](https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/interaction_serial_evaluator.py#L207)  
- [interaction_serial_evaluator.py#L315](https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/interaction_serial_evaluator.py#L315)