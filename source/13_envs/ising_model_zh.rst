Ising Model 
~~~~~~~~~~~~~~~~

概述
=======
伊辛模型是一种经典的物理模型，用于描述铁磁性材料的微观磁态。在本环境中，该模型被扩展为一个多智能体系统，每个智能体通过局部交互影响整体系统的磁态。智能体的目标是通过改变自身的自旋状态来优化整个系统的有序性。


环境介绍
=========

动作空间
----------

在 Ising Model 环境中，每个智能体的动作空间是离散的，并且由两种可能的动作组成：

- 保持自旋状态不变（通常表示为0）。

- 改变自旋状态（通常表示为1）。

使用 gym 环境空间定义则可表示为：

.. code:: python
    
    from gym import spaces

    action_space = gym.spaces.Discrete(2)

状态空间
----------

状态空间由每个智能体的自旋状态和其观察到的邻居智能体的自旋状态组成。每个智能体的自旋状态可以是+1（向上）或-1（向下）。智能体的观察由 `view_sight` 属性定义，该属性决定了智能体可以观察到的邻居范围。

状态空间可以表示为一个二维数组，其中每个元素对应一个智能体的自旋状态。智能体的局部状态由其 `IsingAgentState`` 对象表示，而全局状态由 `IsingWorld`` 对象的 `global_state`` 属性给出。


奖励空间
-----------
在本环境中，奖励是基于智能体自旋状态与邻居自旋状态的一致性。具体来说，奖励计算如下：



- 对于每个智能体 i，计算其邻居的自旋状态的平均值。

- 智能体 i 的奖励是其自旋状态与邻居自旋状态平均值的乘积的负数。

- 奖励的设计鼓励智能体采取行动，以增加系统的总体有序性。

.. code:: python

    # 对于某一个智能体 agent：
    reward = - 0.5 * global_state[agent.state.p_pos] * np.sum(global_state.flatten() * agent.spin_mask)


终止条件
------------
遇到以下任何一种情况，则环境会该认为当前 episode 终止：

- 达到 episode 的最大上限步数（默认设置为200）

- 当全局序参数order_param达到1时，表示系统已经达到完全有序的状态，此时认为环境任务完成，游戏结束。序参数是系统自旋向上和自旋向下的智能体数量差的绝对值，除以总智能体数量。

    - 具体来说，序参数可以定义为系统中自旋向上的粒子数与总粒子数之差，除以总粒子数，即：$$Order Parameter = \frac{N_{up} - N_{down}} {N_{total}}$$
    
    - 其中，$$N_{\text{up}}$$ 是自旋向上的粒子数，$$N_{\text{down}}$$ 是自旋向下的粒子数，$$N_{\text{total}}$$ 是系统中的总粒子数。

DI-zoo 可运行代码示例
=====================

完整的训练配置文件在 `github
link <https://github.com/opendilab/DI-engine/tree/main/dizoo/ising_env/config>`__
内，对于具体的配置文件，例如 `gym_hybrid_ddpg_config.py <https://github.com/opendilab/DI-engine/blob/main/dizoo/ising_env/config/ising_mfq_config.py>`__ ，使用如下命令即可运行：

.. code:: shell

  python3 ./DI-engine/dizoo/ising_env/config/ising_mfq_config.py


基准算法性能
============

-  待补充