高性能RPC框架
===============================

.. toctree::
   :maxdepth: 2

简介
-------------------------------

DI-engine 集成了 `torch.distributed.rpc <https://pytorch.org/docs/stable/rpc.html>`_  模块，torchrpc专注于高性能P2P的RPC通信，其可以自动选择最佳的通信策略（NVLINK，共享内存，RDMA或是TCP），并且支持 `GPU direct RDMA <https://docs.nvidia.com/cuda/gpudirect-rdma/>`_ 技术。
如果您的集群配备有InfiniBand/RoCE网络，则非常建议您使用torchrpc作为DI-engine的消息队列通信实现；即使没有，torchrpc实现的多线程TCP-RPC通信性能也要优于过去DI-engine基于NNG的消息队列实现。


.. image::
    images/rpc.png
    :width: 600
    :align: center


为什么不是NCCL
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
NCCL作为广泛使用的高性能集合通信库，其专注于诸如AllReduce等集合通信范式，但却不擅长进行任意消息类型的高性能P2P通信，而torchrpc填补了这一空白。
下表给出了其和NCCL的一些对比：

============================  ====================  ====================
  对比                           NCCL                torchrpc
============================  ====================  ====================
支持GPU数据的RDMA-P2P            True                        True
支持CPU数据的RDMA-P2P            False                       True
支持传输的对象类型                只支持tensor                任意python对象
适用场景                         集合通讯                    点对点通信
============================  ====================  ====================



使用
-------------------------------

cli-ditask 引入了新的命令行参数：

- --mq_type：引入了 torchrpc:cuda 和 torchrpc:cpu 选项：

    * torchrpc:cuda：使用torchrpc，需要设置device_map，可以使用GPU direct RDMA。
    * torchrpc:cpu：使用torchrpc，但不设置device_map。 GPU数据会被拷贝到内存进行通信。

- --init-method：init_rpc的初始化入口（如果选择torchrpc作为后端则为必填项），同 `torch.distributed.init_process_group <https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.init_rpc>`_ 中的 init-method 参数。

- --local-cuda-devices：设置当前进程可用GPU设备（可选，只有在mq_type为torchrpc:cuda有效，默认为所有可见设备），用逗号分隔的int列表。

- --cuda-device-map：用于设置torchrpc的 `device\_map <https://pytorch.org/tutorials/recipes/cuda_rpc.html>`_，（可选，只有在mq_type为torchrpc:cuda有效，默认会映射全部可用GPU，使得各GPU之间可以P2P），格式如下：

    .. code-block:: bash

        Format:
        <Peer Node ID>_<Local GPU Local Rank>_<Peer GPU Local Rank>,[...]
        example:
        --cuda-device-map=1_0_1,2_0_2,0_0_0


一个具体的例子：

.. code-block:: bash

    export INIT_METHOD="tcp://XX.XX.XX.XX:12345"
    
    # learner位于节点0，使用GPU0，并建立和节点1的GPU0间的P2P映射
    ditask --package . \
        --main atari_dqn_dist_rdma.main \
        --parallel-workers 1 \
        --labels learner \
        --mq-type torchrpc:cuda \
        --init-method=${INIT_METHOD} \
        --cuda-device-map 1_0_0 \
        --node-ids 0

    # collector位于节点1，使用GPU0，并建立和节点0的GPU0间的P2P映射
    ditask --package . \
        --main atari_dqn_dist_rdma.main \
        --parallel-workers 1 \
        --labels collector \
        --mq-type torchrpc \
        --init-method=${INIT_METHOD} \
        --global_rank 1 \
        --attach-to 0_0_0 \
        --node-ids 1

注意事项
-------------------------------

如果您在K8S或其他容器集群环境下使用DI-engine，请使用opendilab提供的镜像，否则torchrpc可能无法正常工作。

多IB设备
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
如果您的集群环境有多张HCA设备，torchrpc可能不会正常工作，这一问题是由torchrpc底层的tensorpip代码缺陷导致的，我们在opendilab的镜像中修复了这一问题。如果您无法使用我们的镜像，请保证您机器上的HCA的端口都处于可用状态。


容器环境
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torchrpc无法判断自己是否处于容器环境，我们的opendilab的镜像增加了容器环境的判断功能，避免torchrpc在容器环境下进行GPU设备映射导致的初始化错误。

和fork()混合使用
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torchrpc没有考虑到进程混合使用RDMA和fork的情况。如果在使用RDMA时没有进行相应的初始化措施，使用fork会出现严重问题， `请参考 <https://www.rdmamojo.com/2012/05/24/ibv_fork_init/>`_。 
因此，如果您在IB/RoCE网络环境下使用torchrpc，请指定环境变量 `IBV_FORK_SAFE=1` 和 `RDMAV_FORK_SAFE=1`。

