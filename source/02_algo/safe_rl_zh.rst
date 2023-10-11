安全强化学习
====================

问题定义和研究动机
---------------------
安全强化学习（Safe Reinforcement Learning），是强化学习的一个细分发展方向，与强化学习一样是一个最大化预期回报的策略学习过程，这一方向主要研究如何在学习和部署过程中既能确保合理的系统性能又能符合安全条件的约束。

在强化学习训练过程中通常会出现以下安全问题：

- 对环境的负面影响（Negative Side Effects）。

- 发掘奖励函数漏洞（Reward Hacking）。
  
- 信息有限时不采取错误动作（Scalable Oversight）。
  
- 安全地探索（Safe Exploration）。
  
- 对新环境或数据分布的安全性（Robustness to Distributional Shift）。

由于这些问题的存在，提出了限制的马尔科夫决策过程（CMDP）和安全强化学习的定义。限制的马尔科夫决策过程（CMDP）由六元组 :math:`(S, A, P, r, c, \mu)`，即状态空间、动作空间、状态转移函数、奖励、代价、代价阈值组成。智能体采取动作后不仅会收到奖励 r 还会得到代价 c ，策略目标是在不超过代价阈值的约束条件下最大化长期奖励：

\ :math:`\max_{\pi}\mathbb{E}_{\tau\sim\pi}\big[R(\tau)\big],\quad s.t.\quad\mathbb{E}_{\tau\sim\pi}\big[C(\tau)\big]\leq\kappa.`


.. image:: images/safe_gym.gif
   :align: center
   :scale: 50 %

上图是 OpenAI 发布的 safety-gym 环境，传统强化学习训练出的最优策略往往以任务为中心，不考虑对环境和自身的影响、不会考虑是否符合人类预期等等。小车（红色）会以最快速度移动到目标地点（绿色圆柱体），完全没有避开地面的陷阱区域（蓝色圆圈），移动路径上如果有障碍物（青色立方体）则会撞开或强行从边缘擦过。

研究方向
--------

当前的安全强化学习领域大致可以分为以下几大方向：

1. 基于策略（Policy-based）

2. 基于模型（Model-based）

3. 离线数据集训练（Offline）

4. 其他方法（Others）

更详细的分类以及对应算法，请参见下图（摘自 `omnisafe <https://github.com/PKU-Alignment/omnisafe>`__ ）：

.. image:: images/safe_rl_registry.png
   :align: center
   :scale: 50 %

上面的分类是与强化学习主体领域的分类一致的，这里针对安全强化学习领域重点介绍两种解决方案的思路：

- 原问题对偶化（Primal Dual）
  
- 原问题（Primal）


原问题对偶化（Primal Dual）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 safe rl 的原问题中，目标和约束都不是凸的，但是可以使用拉格朗日乘子法转换成解对偶问题，对偶问题是极小化凸问题，是可以求解的，这种方案有很多经典工作[]。

拉格朗日函数：:math:`\mathcal{L}(\pi,\lambda)=V(\pi)+\Sigma\lambda_i(U_i(\pi)-c_i),\lambda\geq0`

拉格朗日对偶函数：:math:`d(\lambda)=\max_{\pi\in\mathcal{P}(\mathcal{S})}\mathcal{L}(\pi,\lambda)`

最小化对偶函数：:math:`D^*=\min_{\lambda\in\mathbb{R}_+}d(\lambda)` 就可以得到对偶问题最优解。

原问题（Primal）
~~~~~~~~~~~~~~~~~~~
使用对偶化方案虽然很好地保证了问题的可解性，但是训练迭代的速度很慢，在优化策略函数的同时还要优化对偶函数，同时在选择拉格朗日乘子时并不轻松。因此有些方法不直接关注整个原问题的求解，而去利用 Natural Policy Gradiants 中的单步更新公式：

.. image:: images/safe_rl_npg.png
   :align: center
   :scale: 50 %

在每一步更新时求解一个比较简单的单步约束优化问题，保证每一次更新都不违法约束并提升表现，自然最终会得到一个符合约束的解。代表方法是 CPO 和 PCPO 等。


未来展望
--------

安全强化学习领域的发展时间并不长，而且从定义上来说需要依附于强化学习主领域而存在。在当前主流强化学习的研究领域都可以进一步添加“安全”这一限制条件来进行新的讨论与研究，前景比较宽广。


参考文献
--------

.. [1] Amodei D, Olah C, Steinhardt J, et al. Concrete problems in AI safety[J]. arXiv preprint arXiv:1606.06565, 2016.

.. [2] Paternain, S., Chamon, L. F., Calvo-Fullana, M., & Ribeiro, A. (2019). Constrained reinforcement learning has zero duality gap.arXiv preprint arXiv:1910.13393.

.. [3] Paternain, S., Calvo-Fullana, M., Chamon, L. F., & Ribeiro, A. (2019). Safe policies for reinforcement learning via primal-dual methods.arXiv preprint arXiv:1911.09101.

.. [4] Ding, D., Wei, X., Yang, Z., Wang, Z., & Jovanovic, M. (2021, March). Provably efficient safe exploration via primal-dual policy optimization. InInternational Conference on Artificial Intelligence and Statistics(pp. 3304-3312). PMLR.

.. [5] Ding, D., Zhang, K., Basar, T., & Jovanovic, M. R. (2020). Natural Policy Gradient Primal-Dual Method for Constrained Markov Decision Processes. InNeurIPS.

.. [6] Achiam J, Held D, Tamar A, et al. Constrained policy optimization[C]//International conference on machine learning. PMLR, 2017: 22-31.

.. [7] https://zhuanlan.zhihu.com/p/407168691

.. [8] https://zhuanlan.zhihu.com/p/347272765

.. [9] https://github.com/PKU-Alignment/omnisafe