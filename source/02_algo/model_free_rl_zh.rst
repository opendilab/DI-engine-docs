
无模型的强化学习
##################

问题定义和研究动机
-------------------
在强化学习（RL）领域中，无模型的强化学习算法 (Model-Free Reinforcement Learning, Model-Free RL)是强化学习的一个重要分支。与基于模型的算法相反，无模型的强化学习算法 
是一种不使用马尔科夫决策过程（MDP）中的概率转移分布（和奖励函数）的算法 [14]_（一个RL环境可以用马尔可夫决策过程（MDP）来描述）。状态转移分布和奖励函数通常被统称为环境（或MDP）的 "模型"，因此该类算法被称为 "无模型"算法，即直接通过和环境交互来学习到一个价值函数或者策略函数。

由于现实世界的复杂环境中，往往环境的转移概率，奖赏函数等等是非常难以直接获得的。 
如果智能体想在一个场景下使用模型，那它必须完全从经验中学习，这会带来很多挑战。最大的挑战就是，智能体探索出来的模型和真实模型之间存在误差，而这种误差会导致智能体在学习到的模型中表现很好，但在真实的环境中表现得不好（甚至很差）。基于模型的学习从根本上讲是非常困难的，即使你愿意花费大量的时间和计算力，最终的结果也可能达不到预期的效果。
虽然无模型学习放弃了有模型学习在样本效率方面的潜在收益，但是他们往往更加易于实现和调整 [18]_。 所以相对于有模型的强化学习算法，无模型的强化学习算法在各个领域里得到了更广泛的应用。

研究方向
--------
无模型强化学习算法可以被分类为三个种类：Value-based RL, Policy-based RL, Actor-Critic-based RL

- Value-based RL: DQN [1]_ , C51/QRDQN/IQN/FQF [2]_, R2D2 [3]_, GTrXL [4]_, SQL [5]_

- Actor-Critic-based RL: DDPG [6]_. TD3 [7]_, D4PG [8]_, SAC[9]_, ACER [10]_, PPO [11]_, PPG [12]_, IMPALA [13]_

当然还有很多其他的分类方式，如C51/QRDQN/IQN/FQF对比其他算法可以被分类为值分布（distributional RL）算法与经典强化学习算法。R2D2，IMPALA，D4PG等算法对比其他算法则可被分类为分布式强化学习算法。
在Actor-Critic-based RL 框架中，研究方向也可以被细分成确定策略梯度算法，如DDPG [6]_， TD3 [7]_，和随机策略梯度算法，如SAC [9]_，PPO [11]_。

值分布（distributional RL）算法通过建模动作状态对到一个奖励分布，来替代平均奖励, 在一些atari游戏上取得了更好的效果。分布式强化学习极大地提高了各个部分的效率，并可以解决更复杂的问题， 如星际争霸2（SC2） [15]_ 和 DOTA2 [16]_ 这样超大规模的决策问题。 更多关于分布式强化学习的介绍，请参考 `分布式强化学习 <../02_algo/distributed_rl_zh.html>`_ 。
确定策略梯度算法对比随机策略梯度算法往往需要更少的样本，理论上更加容易收敛。Policy-based RL对比Value-based RL 有如下优劣势 

优势
~~~~~~~
1. 更好的收敛性质
2. 在高维动作空间上有更好的表现，可以解决连续动作空间问题
3. 可以学习到随机策略

劣势
~~~~~
1. 对于policy的evaluation方差过大，导致估计不准确

未来展望
---------
目前无模型的强化学习算法还有很多问题有待探索和解决

1. 提高无模型学习的样本利用效率

2. 如何设计奖励函数让智能体更高效地达到收益最大化

3. 如何更好地平衡学习中的探索（exploration）与利用（exploitation）问题



参考文献
----------

.. [1] Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." nature 518.7540 (2015): 529-533.

.. [2] Bellemare, Marc G., Will Dabney, and Rémi Munos. "A distributional perspective on reinforcement learning." International Conference on Machine Learning. PMLR, 2017.

.. [3] Kapturowski, Steven, et al. "Recurrent experience replay in distributed reinforcement learning." International conference on learning representations. 2018.

.. [4] Parisotto, Emilio, et al. "Stabilizing transformers for reinforcement learning." International conference on machine learning. PMLR, 2020.

.. [5] Haarnoja, Tuomas, et al. "Reinforcement learning with deep energy-based policies." International conference on machine learning. PMLR, 2017.

.. [6] Casas, Noe. "Deep deterministic policy gradient for urban traffic light control." arXiv preprint arXiv:1703.09035 (2017).

.. [7] Fujimoto, Scott, Herke Hoof, and David Meger. "Addressing function approximation error in actor-critic methods." International conference on machine learning. PMLR, 2018.

.. [8] Barth-Maron, Gabriel, et al. "Distributed distributional deterministic policy gradients." arXiv preprint arXiv:1804.08617 (2018).

.. [9] Haarnoja, Tuomas, et al. "Soft actor-critic algorithms and applications." arXiv preprint arXiv:1812.05905 (2018).

.. [10] Sample Efficient Actor-Critic with Experience Replay.

.. [11] Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).

.. [12] Cobbe, Karl W., et al. "Phasic policy gradient." International Conference on Machine Learning. PMLR, 2021.

.. [13] Espeholt, Lasse, et al. "Impala: Scalable distributed deep-rl with importance weighted actor-learner architectures." International conference on machine learning. PMLR, 2018.

.. [14] Montague, P. Read. "Reinforcement learning: an introduction, by Sutton, RS and Barto, AG." Trends in cognitive sciences 3.9 (1999): 360.

.. [15] Oriol Vinyals, Igor Babuschkin, David Silver, et al. Grandmaster level in StarCraft II using multi-agent reinforcement learning. Nat. 575(7782): 350-354 (2019)

.. [16] Christopher Berner, Greg Brockman, et al. Dota 2 with Large Scale Deep Reinforcement Learning. CoRR abs/1912.06680 (2019)

.. [17] RL Course by David Silver https://www.youtube.com/watch?v=KHZVXao4qXs&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&index=7

.. [18] OpenAI Spinning-up https://spinningup.readthedocs.io/zh_CN/latest/spinningup/rl_intro2.html