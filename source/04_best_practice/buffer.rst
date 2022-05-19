Buffer 使用指南
===============================

Buffer 入门
-------------------------------

在 Off-policy 算法中，我们通常需要使用缓存池来实现经验回放（Experience Replay）机制。
DI-engine 提供了 \ **DequeBuffer**\ 来实现一般缓存池的功能，其本质是一个先入先出（FIFO）队列。用户可以通过以下命令创建 DequeBuffer 对象:

.. code-block:: python

    from ding.data import DequeBuffer

    buffer = DequeBuffer(size=10)

接下来，可以直接使用封装好的中间件来调用该 buffer 对象，完成训练中数据的缓存和采样操作（\ **推荐方式**\）。

.. code-block:: python

    task.use(data_pusher(cfg, buffer))
    task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer))


此外，用户也可以通过自定义的方式实现数据的存储与采样，例如：

.. code-block:: python

    # 数据存入，每次处理一条样本。
    # 在 DI-engine 的中间件中，存储数据类型通常为字典，记录样本的 obs，next_obs，action，reward 等信息。
    for _ in range(10):
        buffer.push('a')

    # 数据采样每次处理多条样本，用户需要明确指定采样的数量，replace 表示采样时是否放回，默认值为 False。
    # 采样操作会返回 BufferedData 对象，包含原数据本身，原数据在 buffer 中的索引，以及数据对应的 meta 信息。
    # 例如：BufferedData(data=np.ones([3, 2]), index='67bdfadcd68411ec9e134649caa90281', meta={})
    buffered_data = buffer.sample(3, replace=False)


Buffer 进阶
-------------------------------

在上一节中，我们提供了 buffer 最基本的应用场景。接下来，将为用户展示 buffer 更全面的功能。


**优先级采样**

在一些算法中，需要用到优先级采样。在 DI-engine 中，使用 PriorityExperienceReplay 中间件，可以赋予 buffer 该功能。
同时，在填入样本时，需要在 meta 中补充优先级信息，如下所示。\ **优先级采样会增加采样耗时**\。

.. code-block:: python
    
    from ding.data.buffer.middleware import PriorityExperienceReplay

    buffer = DequeBuffer(size=10)
    buffer.use(PriorityExperienceReplay(buffer, IS_weight=True))
    for _ in range(10):
        # meta 的本质为一个字典，用于补充对样本的描述，默认为空。
        buffer.push(self._data, meta={"priority": 2.0})
    buffered_data = buffer.sample(3)


**样本克隆**

在默认情况下，当 buffer 中存储可变对象（如 list、np.array、torch.tensor 等），采样操作事实上是返回了对该对象的引用。
如果在后续的操作中，该引用的内容发生了修改，样本池中的对应内容也会被修改。
使用 clone_object 中间件，可赋予 buffer 克隆样本的功能，即在采样时返回 buffer 中对象的拷贝，从而保护 buffer 中的内容不被修改。\ **样本克隆会显著增加采样耗时**\。

.. code-block:: python
    
    from ding.data.buffer.middleware import clone_object

    buffer = DequeBuffer(size=10)
    buffer.use(clone_object())


**分组采样**


在某些算法或环境中，收集和存储整个剧集比分离样本更有用。例如：在国际象棋、围棋或纸牌游戏中，玩家只有在游戏结束时才能获得奖励。
一些算法，例如 Hindsight Experience Replay (HER)，必须对一条条完整的 episode 进行采样并对其进行操作。
这时，用户可以使用分组采样的方式实现这一目标。\ **分组采样会严重增加采样耗时**\。

.. code-block:: python

    buffer = DequeBuffer(size=10)

    # 填入数据时，用户需要在 meta 中补充分组信息，如以 "episode" 为分组关键字，值对应为具体的组别
    buffer.push("a", {"episode": 1})
    buffer.push("b", {"episode": 2})
    buffer.push("c", {"episode": 2})

    # 根据关键字 "episode" 来分组，需要保证 buffer 中不同的组的数量不小于采样数量。
    grouped_data = buffer.sample(2, groupby="episode")

之后，可以通过 group_sample 中间件实现样本的后处理工作，如：选择是否打乱组内数据，以及设定每组数据的最大长度。

.. code-block:: python
    
    from ding.data.buffer.middleware import group_sample

    buffer = DequeBuffer(size=10)
    buffer.use(group_sample())


