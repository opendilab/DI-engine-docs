多智能体强化学习
===============================


考虑到在现实场景中通常会同时存在多个智能体（agent），对于强化学习的研究逐渐从单智能体领域延伸到多智能体。在多智能体强化学习（Multi-agent Reinforcement Learning, MARL）中，同时存在多个智能体与环境交互，每个智能体仍然是遵循着强化学习的目标，也就是最大化能够获得的累积回报，而此时环境全局状态（global state）的改变以及回报值（reward）就和所有智能体的联合动作（joint action）相关了。因此在智能体策略学习的过程中，需要考虑联合动作的影响。

.. image:: images/MARL_summary.png
   :align: center
   :scale: 50 %

在该图中，system表示多智能体环境 :math:`Agent_i` 表示第i个智能体，:math:`a_i`表示第i个智能体采取的动作，:math:`r_i`表示第i个智能体获取的局部奖励。
在训练过程中，各个智能体分别与环境进行交互，系统会反馈回联合奖励。


近年来，深度强化学习在多智能体环境和游戏中取得了巨大的成功，通过与环境进行有效的交互，我们可以得到性能卓著的智能体。例如星际争霸StarCraftII的子环境SMAC,足球游戏Gfootball,以及一些自动驾驶环境。未来，MARL还可能被更广泛地应用于资源管理、交通系统等各个领域。

为了方便读者对 MARL 有一个初步的认知，在以下章节，我们将首先讲解多智能体强化学习与单智能体强化学习的区别和面临的挑战：

之后，我们讲解在MARL cooperation任务中的一些常用解决思路：



多智能体强化学习面临的挑战
-------------------------------
环境的不稳定性：智能体在做决策的同时，其他智能体也在采取动作，而环境状态的变化与所有智能体的联合动作相关，这会导致在MARL训练中的非平稳性

智能体获取信息的局限性：在一些环境中（例如SMAC），对于单个智能体而言其不一定能够获得全局的信息，智能体仅能获取局部的观测信息，但无法得知其他智能体的观测信息、动作和奖励等信息；

个体的目标一致性：各智能体的目标可能是最优的全局回报；也可能是各自局部回报的最优；

可拓展性：在大规模的多智能体系统中，就会涉及到高维度的状态空间和动作空间，对于模型表达能力和真实场景中的硬件算力有一定的要求。



MARL cooperation解决思路
------------------------------------
对于MARL cooperation任务来说，最简单的思路就是将单智能体强化学习方法直接套用在多智能体系统中，即每个智能体把其他智能体都当做环境中的因素，仍然按照单智能体学习的方式、通过与环境的交互来更新策略；这是 independent Q-learning， independent PPO方法的思想，但是由于环境的非平稳性和智能体观测的局部性，这些方法很难取得不错的效果。

目前MARL cooperation主要是采用CTDE(centralized training and decentralized execute)的方法，主要有两类解决思路，Valued-based MARL和Actor-Critic MARL。

对于Valued-based MARL，主要的思路是将全局的reward值分解为可以供各个agent学习的局部reward值，从而便于智能体的训练。主要有QMIX，WQMIX，QTRAN等方法。具体可参考MARL RL 算法举例。

对于Actor-critic MARL，主要的思路是学习一个适用于多智能体的策略网络。主要有COMA,MAPPO等方法。具体可参考MARL RL 算法举例。



MARL算法举例
------------------------------------



参考文献
----------

- Rashid, Tabish, et al. "Weighted qmix: Expanding monotonic value function factorisation for deep multi-agent reinforcement learning." arXiv preprint arXiv:2006.10800 (2020).

- Tabish Rashid, Mikayel Samvelyan, Christian Schroeder de Witt, Gregory Farquhar, Jakob Foerster, Shimon Whiteson. Qmix: Monotonic value function factorisation for deep multi-agent reinforcement learning. International Conference on Machine Learning. PMLR, 2018.

- Peter Sunehag, Guy Lever, Audrunas Gruslys, Wojciech Marian Czarnecki, Vinicius Zambaldi, Max Jaderberg, Marc Lanctot, Nicolas Sonnerat, Joel Z. Leibo, Karl Tuyls, Thore Graepel. Value-decomposition networks for cooperative multi-agent learning. arXiv preprint arXiv:1706.05296, 2017.

- Kyunghwan Son, Daewoo Kim, Wan Ju Kang, David Earl Hostallero, Yung Yi. QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning. International Conference on Machine Learning. PMLR, 2019.

- Jakob Foerster, Gregory Farquhar, Triantafyllos Afouras, Nantas Nardelli, Shimon Whiteson. Counterfactual Multi-Agent Policy Gradients. In Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence, 2018.

- Jayesh K. Gupta, Maxim Egorov, Mykel Kochenderfer. Cooperative multi-agent control using deep reinforcement learning. International Conference on Autonomous Agents and Multiagent Systems, 2017.

- Ryan Lowe, Yi Wu, Aviv Tamar, Jean Harb, Pieter Abbeel, Igor Mordatch. Multi-agent actor-critic for mixed cooperative-competitive environments. arXiv preprint arXiv:1706.02275, 2017.

- Mikayel Samvelyan, Tabish Rashid, Christian Schroeder de Witt, Gregory Farquhar, Nantas Nardelli, Tim G. J. Rudner, Chia-Man Hung, Philip H. S. Torr, Jakob Foerster, Shimon Whiteson. The StarCraft Multi-Agent Challenge. arXiv preprint arXiv:1902.04043, 2019.