TREX
^^^^^^^

Overview
---------
TREX (Trajectory-ranked Reward Extrapolation) was first proposed in `Extrapolating Beyond Suboptimal Demonstrations via Inverse Reinforcement Learning from Observations <https://arxiv.org/abs/1904.06387>`_, which uses ranked demonstrations to inference reward function. Different from previous methods, TREX seeks a reward function that explains the ranking over demonstrations, rather than justifying the demonstrations. Hence, this algorithm can learn good policies from highly suboptimal demostrations.

Quick Facts
-------------
1. Demonstrations used in TREX require **ranking** information.

2. The reward function optimization can be viewed as a simple binary classification problem.

3. Usually, multiple data augmentation methods should be applied to prevent reward network from over-fitting.

Key Equations or Key Graphs
---------------------------
The loss for reward function:

.. math::

   \mathcal{L}(\theta)=\mathbb{E}_{\tau_i, \tau_j \sim \Pi}[\xi(\mathcal{P}(\hat J_\theta(\tau_i) < \hat J_\theta(\tau_j)),\tau_i \prec \tau_j)]

The function P is defined as follow:

.. math::

   \mathcal{P}(\hat J_\theta(\tau_i) < \hat J_\theta(\tau_j)) \approx \frac {exp \sum_{s \in \tau_j} \hat r_\theta(s)} {exp \sum_{s \in \tau_i} \hat r_\theta(s) + exp \sum_{s \in \tau_j} \hat r_\theta(s)}

The final loss function is in cross entropy form:

.. math::
   \mathcal{L}(\theta) = - \sum_{\tau_i \prec \tau_j} log \frac {exp \sum_{s \in \tau_j} \hat r_\theta(s)} {exp \sum_{s \in \tau_i} \hat r_\theta(s) + exp \sum_{s \in \tau_j} \hat r_\theta(s)}

Pseudo-code
---------------
.. image:: images/TREX.png
   :align: center
   :scale: 110%

Extensions
----------
TREX can be combined with the following methods：

    - PPO `Proximal Policy Optimization <https://arxiv.org/pdf/1707.06347.pdf>`_

    - SAC `Soft Actor-Critic <https://arxiv.org/pdf/1801.01290>`_

    Given demonstrations generated from RL algorithms or human knowledge (only **observations** and **rankings** are needed), TREX will infer the reward function of the environment. Then the reward function can be applied to RL algorithms like PPO or SAC to estimate rewards while training.

Implementations
----------------
The input of the reward model is observations and its output is the predicted reward value. The default reward model is defined as follows:

.. autoclass:: ding.reward_model.TrexRewardModel
   :noindex:


实验 Benchmark
------------------
+---------------------+-----------------+-----------------------------------------------------+----------------------------------------------------+--------------------------+
| environment         |best mean reward | PPO                                                 |                   TREX+PPO                         |    config link           |
+=====================+=================+=====================================================+====================================================+==========================+
|                     |                 |                                                     |                                                    |`config_link_l <https://  |
|                     |                 |                                                     |                                                    |github.com/opendilab/     |
|                     |                 |                                                     |                                                    |DI-engine/tree/main/dizoo/|
|Lunarlander          |  2M env_step,   |.. image:: images/benchmark/lunar_lander_ppo.png     |.. image:: images/benchmark/lunarlander_ppo_trex.png|box2d/lunarlander/config/ |
|                     |  reward 200     |                                                     |                                                    |lunarlander_trex_dqn_     |
|                     |                 |                                                     |                                                    |config.py>`_              |
+---------------------+-----------------+-----------------------------------------------------+----------------------------------------------------+--------------------------+


Reference
----------

Daniel S. Brown, Wonjoon Goo, Prabhat Nagarajan, Scott Niekum: “Extrapolating Beyond Suboptimal Demonstrations via Inverse Reinforcement Learning from Observations”, 2019; arXiv:1904.06387. https://arxiv.org/abs/1904.06387
