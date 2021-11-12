DQN
^^^^^^^

综述
---------
DQN最初在论文 `Playing Atari with Deep Reinforcement Learning <https://arxiv.org/abs/1312.5602>`_ 中被提出, 它将 Q-learning 与深度神经网络相结合。与以前的传统强化学习方法不同，DQN 使用深度神经网络来估计 Q 值，并通过计算时序差分（TD, Temporal-Difference） 损失，利用梯度下降算法进行更新。

快速了解
-------------
1. DQN 是一个 **model-free** （无模型） 且 **value-based** （基于值函数） 的强化学习算法。

2. DQN 只支持 **离散** 动作空间。

3. DQN 是一个 **off-policy** （离策略） 算法.

4. 通常，DQN 使用 **eps-greedy** （episilon贪婪） 或 **multinomial sample** （多项式采样） 来做 exploration（探索）。

5. DQN + RNN = DRQN

6. DI-engine 中实现的 DQN 支持 **多维度离散** 动作空间（多个离散动作）。

重要公示/重要图示
---------------------------
DQN 中的 TD-loss 是：

.. math::

   L(w)=\mathbb{E}\left[(\underbrace{r+\gamma \max _{a^{\prime}} Q\left(s^{\prime}, a^{\prime}, w\right)}_{\text {Target }}-Q(s, a, w))^{2}\right]

伪代码
---------------
.. image:: images/DQN.png
   :align: center
   :scale: 55%

.. note::
   与 vanilla 版本相比，DQN 在算法和实现方面都得到了显著改进。在算法部分，**n-step TD-loss, PER, target network and dueling head** 被广泛使用。对于实施细节，根据环境步数（envstep，policy 与环境的交互次数），在训练期间，epsilon 从一个较高的初始值退火到零，而不是保持不变。

扩展
-----------
DQN 可以和以下相结合：

    - PER (Prioritized Experience Replay，带优先级的经验回放池)

        `PRIORITIZED EXPERIENCE REPLAY <https://arxiv.org/abs/1511.05952>`_ 用一种特殊定义的“优先”来代替经验回放池中的均匀采样，该采样由各种指标定义，如绝对TD误差、观察的新颖性等。通过优先采样，DQN的收敛速度和性能可以得到很大的提高。

        PER 的一种实现可以这样描述：

        .. image:: images/PERDQN.png
           :align: center
           :scale: 60%

    - 多步（Multi-step） TD-loss

        .. note::
           在单步 TD-loss 中，Q-learning 通过贝尔曼更新 :math:`r(s,a)+\gamma \mathop{max}\limits_{a^*}Q(s',a^*)` 学习 :math:`Q(s,a)` 。而在 n步 TD-loss 中，方程是 :math:`\sum_{t=0}^{n-1}\gamma^t r(s_t,a_t) + \gamma^n \mathop{max}\limits_{a^*}Q(s_n,a^*)` 。关于 n-step Q-learning 的问题是， 当采用 epsilon 贪婪时， q 值估计值是有偏的， 因为 :math:`r(s_t,a_t)` 中 t >= 1 时是在 epsilon-greedy 下采样的，而不睡从策略本身。然而，实际上multi-step TD-loss 与 epsilon-greedy 结合使用，一般可以提升 DQN 效果。

    - 目标网络（target network）/ 双 DQN （Double DQN）

      双 DQN, 在 `Deep Reinforcement Learning with Double Q-learning <https://arxiv.org/abs/1509.06461>`_ 中被提出，是 DQN 的一种常见变种。此方法维护另一个 Q 网络，称为目标网络，该网络由当前网络按固定频率（时间间隔）更新。

      双 DQN 不会选择当前网络中离散动作空间中的最大q值，而是 **首先查找当前网络中q值最大的动作，然后根据该动作从目标网络获取q值**。该变种可以解决q值过高估计的问题，减少向上的偏差。

        .. note::
            过高估计可能是由函数近似误差（q值表的神经网络）、环境噪声、数值不稳定等原因造成的。

    - Dueling head

      Dueling head 结构用于实现每个动作的状态-价值和优势的分解，并利用这两个部分构建最终的q值，从而更好地评估一些与动作选择无关的状态的价值。下图进行了示意：

        .. image:: images/Dueling_DQN.png
           :align: center
           :height: 300

    - RNN (DRQN, R2D2)

实现
----------------
DQNPolicy 的默认 config 如下所示：

.. autoclass:: ding.policy.dqn.DQNPolicy
   :noindex:

其中使用的网络接口如下所示：

.. autoclass:: ding.model.template.q_learning.DQN
   :members: __init__, forward
   :noindex:

Benchmark
------------------

（在5个不同的随机种子，10M个env_step下重复实验得到）

+------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
| environment      |best mean reward | image                                               | config link              | comparison           |
+==================+=================+=====================================================+==========================+======================+
|                  |                 |                                                     | `config link <\          |                      |
|                  |                 |                                                     | https://github.com/op\   |                      |
|                  |                 |                                                     | endilab/DI-engine/tr\    |                      |
|                  |                 |                                                     | ee/main/\                |  tianshou(20)        |
| pong             |  20             |.. image:: images/benchmark/pong_dqn.png             | dizoo\                   |                      |
|                  |                 |                                                     | /atari/config/serial/p\  |  Sb3(20)             |
|                  |                 |                                                     | ong/pong_dqn_config.py>`_|                      |
+------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                  |                 |                                                     | `config link <\          |                      |
|                  |                 |                                                     | https://github.com/op\   |  tianshou(7307)      |
|                  |                 |                                                     | endilab/DI-engine/tr\    |                      |
|                  |                 |                                                     | ee/main/\                |  Rllib(7968)         |
| qbert            |  17866          |.. image:: images/benchmark/qbert_dqn.png            | dizoo/a\                 |                      |
|                  |                 |                                                     | tari/config/serial/qbe\  |                      |
|                  |                 |                                                     | rt/qbert_dqn_config.py>`_|  Sb3(9496)           |
+------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                  |                 |                                                     | `config link <\          |                      |
|                  |                 |                                                     | https://github.com/op\   |  tianshou(812)       |
|                  |                 |                                                     | endilab/DI-engine/tr\    |                      |
|                  |                 |                                                     | ee/main/\                |  Rllib(1001)         |
| spaceinvaders    | 1880            |.. image:: images/benchmark/spaceinvaders_dqn.png    | d\                       |                      |
|                  |                 |                                                     | izoo/atari/config/seri\  |                      |
|                  |                 |                                                     | al/spaceinvaders/space\  |  Sb3(622)            |
|                  |                 |                                                     | invaders_dqn_config.py>`_|                      |
+------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+



参考文献
----------

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller: “Playing Atari with Deep Reinforcement Learning”, 2013; arXiv:1312.5602. https://arxiv.org/abs/1312.5602
