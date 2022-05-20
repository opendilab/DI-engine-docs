Buffer 使用指南
===============================

Buffer 入门
-------------------------------

在 Off-policy RL 算法中，通常会使用经验回放（Experience Replay）机制来提高样本利用效率并降低样本之间的相关性。为此需要定义一个缓存池来实现样本的存储，处理和采样。
DI-engine 提供了 \ **DequeBuffer** \ 来实现缓存池的常见功能，其本质是一个先入先出（FIFO）队列。用户可以通过以下命令创建 DequeBuffer 对象:

.. code-block:: python

    from ding.data import DequeBuffer

    buffer = DequeBuffer(size=10)

接下来，用户可以通过 DI-engine 中封装好的中间件来调用该 buffer 对象，完成训练任务（\ **推荐方式**\）。

.. code-block:: python

    task.use(data_pusher(cfg, buffer))
    task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer))


此外，当用户有其它使用 buffer 缓存数据的需求时，也可以通过自定义的方式实现数据的存储与采样，例如：

.. code-block:: python

    # 数据存入，每次处理一条样本。
    # 在 DI-engine 的中间件中，存储数据类型通常为字典，记录样本的 obs，next_obs，action，reward 等信息。
    for _ in range(10):
        # 读入字符数据 'a'
        buffer.push('a')

    # 数据采样每次处理多条样本，用户需要明确指定采样的数量，replace 表示采样时是否放回，默认值为 False。
    # 采样操作会返回 BufferedData 对象，包含原数据本身（data），原数据在 buffer 中的索引（index），以及数据对应的元信息（meta）。
    # 例如：BufferedData(data=np.ones([3, 2]), index='67bdfadcd68411ec9e134649caa90281', meta={})
    buffered_data = buffer.sample(3, replace=False)


Buffer 进阶
-------------------------------

在上一节中，我们提供了 buffer 最基本的应用场景。接下来，将为用户深入展示 buffer 更全面的功能。


**优先级采样**

在一些算法中，需要用到优先级采样。在 DI-engine 中，使用 PriorityExperienceReplay 中间件，可以赋予 buffer 该功能。
如果使用了优先级采样，必须在存入样本时在 meta 中补充优先级信息，如下所示。\ **优先级采样会增加采样耗时**\。

.. code-block:: python
    
    from ding.data.buffer.middleware import PriorityExperienceReplay

    buffer = DequeBuffer(size=10)
    buffer.use(PriorityExperienceReplay(buffer, IS_weight=True))
    for _ in range(10):
        # meta 的本质为一个字典，用于补充对样本的描述，默认为空。
        buffer.push('a', meta={"priority": 2.0})
    buffered_data = buffer.sample(3)


**样本克隆**

在默认情况下，当 buffer 中存储可变对象（如 list、np.array、torch.tensor 等），采样操作事实上是返回了对该对象的引用。
后续对该引用的内容的修改操作，可能会导致样本池中的对应内容也发生变化。
在某些应用场景上，用户可能期望样本池中的数据保持不变，这时就可以使用 clone_object 中间件，在采样时返回 buffer 中对象的拷贝。
这样一来，对拷贝内容的修改就不会影响到 buffer 中的原数据。\ **样本克隆会显著增加采样耗时**\。

.. code-block:: python
    
    from ding.data.buffer.middleware import clone_object

    buffer = DequeBuffer(size=10)
    buffer.use(clone_object())


**分组采样**

在某些特殊环境或算法中，用户可能希望以整个剧集 (episode) 为单位收集，存储，处理样本。
例如：在国际象棋、围棋或纸牌游戏中，玩家只有在游戏结束时才能获得奖励，解决这类任务时算法往往希望对整局游戏进行相关处理，此外像 Hindsight Experience Replay (HER) 等一些算法需要采样完整的 episode，并以 episode 为单位进行相关处理。
这时，用户可以使用分组采样的方式实现这一目标。
存储样本时，用户可以在 meta 补充 "episode" 信息，以明确样本所属的 episode。采样时，通过设定 groupby="episode" 即可来实现以 episode 为关键字的分组采样。\ **分组采样会严重增加采样耗时**\。

.. code-block:: python

    buffer = DequeBuffer(size=10)

    # 填入数据时，用户需要在 meta 中补充分组信息，如以 "episode" 为分组关键字，值对应为具体的组别
    buffer.push("a", {"episode": 1})
    buffer.push("b", {"episode": 2})
    buffer.push("c", {"episode": 2})

    # 根据关键字 "episode" 来分组，需要保证 buffer 中不同的组的数量不小于采样数量。
    grouped_data = buffer.sample(2, groupby="episode")

(可选)另外，在分组采样的基础上，还通过 group_sample 中间件实现样本的后处理工作，如：选择是否打乱同组内数据，以及设定每组数据的最大长度。

.. code-block:: python
    
    from ding.data.buffer.middleware import group_sample

    buffer = DequeBuffer(size=10)
    buffer.use(group_sample())


