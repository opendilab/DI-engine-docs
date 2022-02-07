PPO
^^^^^^^

Overview
---------
PPO(Proximal Policy Optimization) was proposed in `Proximal Policy Optimization Algorithms <https://arxiv.org/pdf/1707.06347.pdf>`_. PPO follows the idea of TRPO, which restricts the step of policy update by KL-divergence, and uses clipped probability ratios of the new and old policies to replace the direct KL-divergence restriction. This adaptation is simpler to implement and avoid the calculation of the Hessian matrix in TRPO.

Quick Facts
-----------
1. PPO is a **model-free** and **policy-based** RL algorithm.

2. PPO supports both **discrete** and **continuous action spaces**.

3. PPO supports **off-policy** mode and **on-policy** mode.

4. PPO can be equipped with RNN.

5. PPO on-policy implementation use double loop(epoch loop and minibatch loop).

Key Equations or Key Graphs
------------------------------
PPO use clipped probability ratios in the policy gradient to prevent the policy from too rapid changes:

.. math::

    L^{C L I P}(\theta)=\hat{\mathbb{E}}_{t}\left[\min \left(r_{t}(\theta) \hat{A}_{t}, \operatorname{clip}\left(r_{t}(\theta), 1-\epsilon, 1+\epsilon\right) \hat{A}_{t}\right)\right]
    L^{C L I P}(\theta)=\hat{E}{t}\left[\min \left(r{t}(\theta) \hat{A}{t}, \operatorname{clip}\left(r{t}(\theta), 1-\epsilon, 1+\epsilon\right) \hat{A}_{t}\right)\right]

with the probability ratio :math:`r_t(\theta)` defined as:

.. math::

    r_{t}(\theta)=\frac{\pi_{\theta}\left(a_{t} \mid s_{t}\right)}{\pi_{\theta_{\text {old }}}\left(a_{t} \mid s_{t}\right)}

When :math:`\hat{A}_t > 0`, :math:`r_t(\theta) > 1 + \epsilon` will be clipped. While when :math:`\hat{A}_t < 0`, :math:`r_t(\theta) < 1 - \epsilon` will be clipped. However, in the paper `Mastering Complex Control in MOBA Games with Deep Reinforcement Learning <https://arxiv.org/abs/1912.09729>`_, the authors claim that when :math:`\hat{A}_t < 0`, a too large :math:`r_t(\theta)` should also be clipped, which introduces dual clip:

.. math::

    \max \left(\min \left(r_{t}(\theta) \hat{A}_{t}, \operatorname{clip}\left(r_{t}(\theta), 1-\epsilon, 1+\epsilon\right) \hat{A}_{t}\right), c \hat{A}_{t}\right)


Pseudo-code
-----------
.. image:: images/PPO.png
   :align: center
   :width: 700

.. note::
   This is the on-policy version of PPO.

Extensions
-----------
PPO can be combined with:
    - Multi-step learning
    - RNN
    - GAE

    .. note::
      Indeed, the standard implementation of PPO contains the many additional optimizations which are not described in the paper. Further details can be found in `IMPLEMENTATION MATTERS IN DEEP POLICY GRADIENTS: A CASE STUDY ON PPO AND TRPO <https://arxiv.org/abs/2005.12729>`_.

Implementation
-----------------
The default config is defined as follows:

    .. autoclass:: ding.policy.ppo.PPOPolicy


    .. autoclass:: ding.model.template.vac.VAC
        :members: forward, compute_actor, compute_critic, compute_actor_critic
        :noindex:


The policy gradient and value update of PPO is implemented as follows:

.. code:: python

    def ppo_error(
            data: namedtuple,
            clip_ratio: float = 0.2,
            use_value_clip: bool = True,
            dual_clip: Optional[float] = None
    ) -> Tuple[namedtuple, namedtuple]:

        assert dual_clip is None or dual_clip > 1.0, "dual_clip value must be greater than 1.0, but get value: {}".format(
            dual_clip
        )
        logit_new, logit_old, action, value_new, value_old, adv, return_, weight = data
        policy_data = ppo_policy_data(logit_new, logit_old, action, adv, weight)
        policy_output, policy_info = ppo_policy_error(policy_data, clip_ratio, dual_clip)
        value_data = ppo_value_data(value_new, value_old, return_, weight)
        value_loss = ppo_value_error(value_data, clip_ratio, use_value_clip)

        return ppo_loss(policy_output.policy_loss, value_loss, policy_output.entropy_loss), policy_info

Some concrete implementation details:

- Recompute advantage: recompute the advantage of historical transitions before the beginning of each training epoch, to keep the estimation
  of advantage close to current policy.

..
For how we compute advantage,

- Value/advantage normalization: we standardize the targets of the value function by using running estimates of the average and standard deviation of the value targets.
  For more implementation details about, users can refer to this discussion
  `<https://github.com/opendilab/DI-engine/discussions/172#discussioncomment-1901038>`_.

..
The Benchmark result of PPO implemented in DI-engine is shown in `Benchmark <../feature/algorithm_overview.html>`_.

Benchmark
-----------

off policy PPO Benchmark:


+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
| environment         |best mean reward | evaluation results                                  | config link              | comparison           |
+=====================+=================+=====================================================+==========================+======================+
|                     |                 |                                                     |`config_link_p <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|                     |                 |                                                     |DI-engine/tree/main/dizoo/|                      |
|Pong                 |  20             |.. image:: images/benchmark/pong_offppo.png          |atari/config/serial/      |                      |
|                     |                 |                                                     |pong/pong_offppo_config   |                      |
|(PongNoFrameskip-v4) |                 |                                                     |.py>`_                    |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_q <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|Qbert                |                 |                                                     |DI-engine/tree/main/dizoo/|                      |
|                     |  16400          |.. image:: images/benchmark/qbert_offppo.png         |atari/config/serial/      |                      |
|(QbertNoFrameskip-v4)|                 |                                                     |qbert/qbert_offppo_config |                      |
|                     |                 |                                                     |.py>`_                    |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_s <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|SpaceInvaders        |                 |                                                     |DI-engine/tree/main/dizoo/|                      |
|                     |  1200           |.. image:: images/benchmark/spaceinvaders_offppo.png |atari/config/serial/      |                      |
|(SpaceInvadersNoFrame|                 |                                                     |spaceinvaders/spaceinva   |                      |
|skip-v4)             |                 |                                                     |ders_offppo_config.py>`_  |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_ho <https:// |                      |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|                     |                 |                                                     |DI-engine/tree/main/dizoo/|                      |
|Hopper               |  300            |.. image:: images/benchmark/hopper_offppo.png        |mujoco/config/serial/ho   |                      |
|                     |                 |                                                     |pper/hopper_offppo_config |                      |
|(Hopper-v3)          |                 |                                                     |.py>`_                    |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_w <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|Walker2d             |                 |                                                     |DI-engine/tree/main/dizoo/|                      |
|                     |  500            |.. image:: images/benchmark/walker2d_offppo.png      |mujoco/config/serial/     |                      |
|(Walker2d-v3)        |                 |                                                     |walker2d/walker2d_        |                      |
|                     |                 |                                                     |offppo_config.py>`_       |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_ha <https:// |                      |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|Halfcheetah          |                 |                                                     |DI-engine/tree/main/dizoo/|                      |
|                     |  2000           |.. image:: images/benchmark/halfcheetah_offppo.png   |mujoco/config/serial/     |                      |
|(Halfcheetah-v3)    |                 |                                                     |halfcheetah/halfcheetah   |                      |
|                     |                 |                                                     |_offppo_config.py>`_      |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+


on policy PPO Benchmark:


+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
| environment         |best mean reward | evaluation results                                  | config link              | comparison           |
+=====================+=================+=====================================================+==========================+======================+
|                     |                 |                                                     |`config_link_p <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|                     |                 |                                                     |DI-engine/tree/main/dizoo/|    RLlib(20)         |
|Pong                 |  20             |.. image:: images/benchmark/pong_onppo.png           |atari/config/serial/      |                      |
|                     |                 |                                                     |pong/pong_onppo_config    |                      |
|(PongNoFrameskip-v4) |                 |                                                     |.py>`_                    |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_q <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|Qbert                |                 |                                                     |DI-engine/tree/main/dizoo/|    RLlib(11085)      |
|                     |  10000          |.. image:: images/benchmark/qbert_onppo.png          |atari/config/serial/      |                      |
|(QbertNoFrameskip-v4)|                 |                                                     |qbert/qbert_onppo_config  |                      |
|                     |                 |                                                     |.py>`_                    |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_s <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|SpaceInvaders        |                 |                                                     |DI-engine/tree/main/dizoo/|    RLlib(671)        |
|                     |  400            |.. image:: images/benchmark/spaceinvaders_onppo.png  |atari/config/serial/      |                      |
|(SpaceInvadersNoFrame|                 |                                                     |spaceinvaders/spacein     |                      |
|skip-v4)             |                 |                                                     |vaders_onppo_config.py>`_ |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_ho <https:// |                      |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|                     |                 |                                                     |DI-engine/tree/main/dizoo/|   Tianshou(2700)     |
|Hopper               |  3000           |.. image:: images/benchmark/hopper_onppo.png         |mujoco/config/serial/     |      Sb3(1567)       |
|                     |                 |                                                     |hopper/hopper_onppo_config|    spinningup(2500)  |
|(Hopper-v3)          |                 |                                                     |.py>`_                    |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_w <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|Walker2d             |                 |                                                     |DI-engine/tree/main/dizoo/|   Tianshou(4500)     |
|                     |  3000           |.. image:: images/benchmark/walker2d_onppo.png       |mujoco/config/serial/     |     Sb3(1230)        |
|(Walker2d-v3)        |                 |                                                     |walker2d/walker2d_        |    spinningup(2500)  |
|                     |                 |                                                     |onppo_config.py>`_        |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_ha <https:// |                      |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|Halfcheetah          |                 |                                                     |DI-engine/tree/main/dizoo/|   Tianshou(7194)     |
|                     |  3500           |.. image:: images/benchmark/halfcheetah_onppo.png    |mujoco/config/serial/     |     Sb3(1976)        |
|(Halfcheetah-v3)     |                 |                                                     |halfcheetah/halfcheetah   |   spinningup(3000)   |
|                     |                 |                                                     |_onppo_config.py>`_       |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+


References
-----------

- John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov: “Proximal Policy Optimization Algorithms”, 2017; [http://arxiv.org/abs/1707.06347 arXiv:1707.06347].

- Logan Engstrom, Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Firdaus Janoos, Larry Rudolph, Aleksander Madry: “Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO”, 2020; [http://arxiv.org/abs/2005.12729 arXiv:2005.12729].

- Andrychowicz M, Raichuk A, Stańczyk P, et al. What matters in on-policy reinforcement learning? a large-scale empirical study[J]. arXiv preprint arXiv:2006.05990, 2020.

- Ye D, Liu Z, Sun M, et al. Mastering complex control in moba games with deep reinforcement learning[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2020, 34(04): 6672-6679.
