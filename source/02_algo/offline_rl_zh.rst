离线强化学习
===============================

引言
-----------------------

离线强化学习（Offline Reinforcement Learning, Offline RL），又称作批量强化学习（Batch Reinforcement Learning, BRL），是强化学习的一种变体，主要研究的是如何利用预先收集的大规模静态数据集来训练强化学习智能体。
利用静态数据集意味着在智能体的训练过程中，Offline RL 不进行任何形式的在线交互与探索，这也是它和一般强化学习最显著的区别。

.. image:: images/offline_no_words.png
   :align: center
   :scale: 50 %

在该图中，(a)是标准的 On-policy RL，智能体使用自身当前策略 :math:`\pi_k` 与环境进行交互而产生交互数据来更新。
(b)是 Off-policy RL，它会将历史的每个策略 :math:`\pi_k` 与环境交互的数据存储在经验池 :math:`\mathcal{D}` 中，即 :math:`\mathcal{D}` 中包含了策略 :math:`\pi_0, \pi_1, ..., \pi_k` 的数据，继而所有的数据都会被用于更新 :math:`\pi_{k+1}`。
(c)是 Offline RL，相比之下，它所使用的数据集 :math:`\mathcal{D}` 中的数据来自某种（可能未知）的行为策略 :math:`\pi_{\beta}` 。该数据集 :math:`\mathcal{D}` 是提前一次性收集好的，不会在训练过程中发生改变，因此这使得使用大规模数据集成为可能。
在训练过程中，智能体不会与环境进行交互，策略只会在学习完成后才会进行与评估与应用。

为什么选择 Offline RL？
-----------------------

Offline RL成为了最近的研究热点，具体原因可以归结为两方面：

第一方面是 Offline RL 本身的优势。深度强化学习在模拟任务和游戏中已经取得了巨大的成功，通过与环境进行有效的交互，我们可以得到性能卓著的智能体。
然而，在现实环境中通过反复试验来训练一般的强化学习策略是昂贵的，在大多数现实世界的问题中甚至是极度危险的，例如自动驾驶和机器人操作。
Offline RL 恰好是在没有任何额外探索的情况下，通过深入研究固定数据集来寻求一种可行的解决方案来减轻潜在风险和成本。
另外，在过去十余年中，随着数据驱动学习方法的出现以及其在机器学习领域的成功，我们知道使用更多数据能获得更好的训练效果。相比于一般的强化学习，能够利用大规模静态数据集的特点是也 Offline RL 的一大优势。

第二方面，经典强化学习算法在离线设定下学习效果往往非常差，学到的策略无法在实际部署中取得令人满意的表现（具体原因见后文）。因此，对这一领域的研究充满挑战的。

Offline RL 概念
------------------------------------

Offline RL 的训练过程
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在训练阶段，离线 Offline RL 要求智能体不能和环境进行交互。 
在这个设定下, 我们先根据行为策略 :math:`\pi_{\beta}(\mathbf{a}\mid \mathbf{s})` 来收集数据得到数据集 :math:`\mathcal{D}`，然后
再利用该数据集训练智能体。以演员-评论家 （actor-critic） 范式为例，给定数据集 :math:`\mathcal{D} = \left\{ (\mathbf{s}, \mathbf{a}, r, \mathbf{s}^{\prime})\right\}`, 
我们可以将价值（value） 的迭代和策略优化表示为:

.. math::
   \hat{Q}^{k+1} \leftarrow \arg\min_{Q} \mathbb{E}_{\mathbf{s}, \mathbf{a} \sim \mathcal{D}} \left[ \left(\hat{\mathcal{B}}^\pi \hat{Q}(\mathbf{s}, \mathbf{a})  - Q(\mathbf{s}, \mathbf{a}) \right)^2 \right],
   \\
   \hat{\pi}^{k+1} \leftarrow \arg\max_{\pi} \mathbb{E}_{\mathbf{s} \sim \mathcal{D}, \mathbf{a} \sim \pi^{k}(\mathbf{a} \mid \mathbf{s})}\left[\hat{Q}^{k+1}(\mathbf{s}, \mathbf{a})\right],

其中， :math:`\hat{\mathcal{B}}^\pi` 表示遵循策略 :math:`\hat{\pi} \left(\mathbf{a} \mid \mathbf{s}\right)` 的贝尔曼操作符, :math:`\hat{\mathcal{B}}^\pi \hat{Q}\left(\mathbf{s}, \mathbf{a}\right) = \mathbb{E}_{\mathbf{s}, \mathbf{a}, \mathbf{s}^{\prime} \sim \mathcal{D}}[ r(\mathbf{s}, \mathbf{a})+\gamma \mathbb{E}_{\mathbf{a}^{\prime} \sim \hat{\pi}^{k}\left(\mathbf{a}^{\prime} \mid \mathbf{s}^{\prime}\right)}\left[\hat{Q}^{k}\left(\mathbf{s}^{\prime}, \mathbf{a}^{\prime}\right)\right] ]`

Offline RL VS 模仿学习
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

模仿学习（Imitation Learning，IL）也使用静态数据进行训练，且在训练过程中不进行探索，这一点上和 Offline RL 是非常相似的。然而，两者也有很多不同之处：

-  目前为止，绝大多数 Offline RL 算法都建立在标准的 Off-Policy RL 算法之上，这些算法倾向于优化某种形式的贝尔曼方程或时间差分误差；而 IL 更符合普通监督学习的范式。
-  大多数 IL 问题假设有一个最优的或一个高性能的专家来提供数据；而 Offline RL 可能需要从大量次优数据中进行学习。
-  大多数 IL 问题没有奖励（reward）的概念；而 Offline RL 需要显式考虑奖励项。
-  一些 IL 问题要求数据被标记为专家经验和非专家经验，而 Offline RL 不做这样的数据区分。


Offline RL VS Off-policy RL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Off-policy RL 通常指能够允许产生训练样本的策略（与环境交互的策略）与当前待优化策略不同的一类 RL 算法。Q-learning 算法、利用Q-函数的 Actor-Critic 算法，以及许多基于模型的强化学习算法（Model-based RL）都属于 Off-policy RL。
然而，Off-policy RL 在学习过程中仍然经常使用额外的交互（即在线数据收集）。


经典强化学习算法在离线设定下的失败
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

很多前人的研究工作都表明经典强化学习算法在 Offline RL 场景表现不佳，甚至很差。Scott Fujimoto 在其论文 [6] 中表明这是因为在这种情况下，策略倾向于选择偏离数据集 :math:`\mathcal{D}` 的动作（out-of-distribution, OOD）。
以基于Q-函数的经典算法为例，当待预估数据与离线训练数据分布相同时，Q-函数的估计才是准确的。
当智能体进行在线探索时，数据集随着策略的更新也不断在更新，策略的马尔科夫静态状态分布和数据集中实际的状态分布始终是一致或者相似的（取决于 on-policy 还是 off-policy）。
但在 Offline RL 场景下，策略的马尔科夫静态状态分布相比原数据集会产生偏移（distributional shift）。如果Q-函数高估了这些训练数据中未曾见过的 :math:`(\mathbf{s}, \mathbf{a})` 对，那么在实际交互中，当智能体选择最大化期望奖励的动作时，便可能选到实际收益非常差的动作，导致整体的表现非常差。


Offline RL 算法分类
------------------------------------

遵循 Aviral Kumar 与 Sergey Levine 在 NeurIPS 2020 Tutorial [1] 中的分类方式，我们将现有的（Model-free） Offline RL 算法分为三类：策略约束方法，基于不确定性的方法和值函数的正则化方法。


策略约束方法
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

该方法核心思想是让待优化策略 :math:`\pi(\mathbf{a} \mid \mathbf{s})` 和行为策略 :math:`\pi_{\beta}(\mathbf{a} \mid \mathbf{s})` 足够接近，以此来保证Q-函数的估计有效。
如 BCQ [6] 提出训练一个生成模型来模拟可能来自离线数据的动作，以及进一步干扰动作的扰动模型（Perturbation network）对生成的动作进行调优。


基于不确定性的方法
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

该方法的目的是尽可能使得Q-函数的估计能够免疫 OOD 动作的影响。首先需要给定不确定性 :math:`\mathbf{U}` 的估计，使得动作的 OOD 程度越高，不确定性越大。
然后将其作为Q-函数的一个惩罚项，就能在一定程度上避免策略选择 OOD 动作。具体做法可参考 [7]。


值函数的正则化方法
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

该方法的优势是不必显式地考虑行为策略的分布。如 CQL [8] 算法提出在优化目标上额外增加一项 :math:`\mathbf{Q}` 的正则项，如下式第一项，

.. math::
   Q_{CQL}(\mathbf{s}, \mathbf{a}) = \arg\min_{Q}\max_{\mu} \mathbb{E}_{\mathbf{s} \sim \mathcal{D}}\mathbb{E}_{\mathbf{a} \sim \mu}[\mathbf{Q}(\mathbf{s}, \mathbf{a})] + \frac{1}{2\alpha}\{ \mathbb{\ Bellman \ Term \ } \}

这一项的作用是在当得到策略 :math:`\mu` 后，会打压该策略中较大的 :math:`\mathbf{Q}` 值，从而避免过估计问题。
 


参考文献
----------

1. Offline (Batch) Reinforcement Learning: A Review of Literature and Applications
2. Levine Sergey, et al. "Offline reinforcement learning: Tutorial, review, and perspectives on open problems." arXiv preprint arXiv:2005.01643 (2020).
3. Agarwal, Rishabh, Dale Schuurmans, and Mohammad Norouzi. "An optimistic perspective on offline reinforcement learning." ICML, 2020.
4. Gulcehre, Caglar, et al. "Rl unplugged: Benchmarks for offline reinforcement learning." Neurips, 2020.
5. Fu, Justin, et al. "D4rl: Datasets for deep data-driven reinforcement learning." arXiv preprint arXiv:2004.07219 (2020).
6. Fujimoto, S., Meger, D., and Precup, D. (2018). Off-policy deep reinforcement learning without exploration. arXiv preprint arXiv:1812.02900.
7. O’Donoghue, B., Osband, I., Munos, R., and Mnih, V. (2018). The uncertainty bellman equation and exploration. In International Conference on Machine Learning, pages 3836–3845.
8. Kumar, A., Zhou, A., Tucker, G., and Levine, S. (2020b). Conservative q-learning for ofﬂine reinforcement learning. In Neural Information Processing Systems (NeurIPS).