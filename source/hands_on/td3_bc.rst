TD3BC
^^^^^^^

Overview
---------

TD3BC, proposed in the 2021 paper `A Minimalist Approach to Offline Reinforcement Learning <https://arxiv.org/abs/2106.06860>`_,
is a simple approach to offline RL where only two changes are made to TD3: a weighted behavior cloning loss is added to the policy update and the states are normalized.
Unlike competing methods there are no changes to architecture or underlying hyperparameters.
The resulting algorithm is a simple to implement and tune baseline, while more than halving the overall run time by removing the additional computational overhead of previous methods.

Quick Facts
-----------
1. TD3BC is an **offline** RL algorithm.

2. TD3BC is based on **TD3** and **behavior cloning**.

Key Equations or Key Graphs
---------------------------
TD3BC simply consists to add a behavior cloning term to TD3 in order to regularize the policy:

.. math::
    \begin{aligned}
    \pi = \arg\max_{\pi} \mathbb{E}_{(s, a) \sim D} [ \lambda Q(s, \pi(s)) - (\pi(s)-a)^2 ]
    \end{aligned}

Additionally, all the states in the dataset are normalized, such that they have mean 0 and standard deviation 1.
This normalization improves the stability of the learned policy.

Pseudocode
----------

.. math::

    :nowrap:

    \begin{algorithm}[H]
        \caption{TD3BC}
        \label{alg1}
    \begin{algorithmic}[1]
        \STATE Input: initial policy parameters $\theta$, Q-function parameters $\phi_1$, $\phi_2$, offline dataset $\mathcal{D}$
        \STATE Set target parameters equal to main parameters $\theta_{\text{targ}} \leftarrow \theta$, $\phi_{\text{targ},1} \leftarrow \phi_1$, $\phi_{\text{targ},2} \leftarrow \phi_2$
        \REPEAT
            \FOR{$j$ in range(however many updates)}
                \STATE Randomly sample a batch of transitions, $B = \{ (s,a,r,s',d) \}$ from $\mathcal{D}$
                \STATE Compute target actions
                \begin{equation*}
                    a'(s') = \text{clip}\left(\mu_{\theta_{\text{targ}}}(s') + \text{clip}(\epsilon,-c,c), a_{Low}, a_{High}\right), \;\;\;\;\; \epsilon \sim \mathcal{N}(0, \sigma)
                \end{equation*}
                \STATE Compute targets
                \begin{equation*}
                    y(r,s',d) = r + \gamma (1-d) \min_{i=1,2} Q_{\phi_{\text{targ},i}}(s', a'(s'))
                \end{equation*}
                \STATE Update Q-functions by one step of gradient descent using
                \begin{align*}
                    & \nabla_{\phi_i} \frac{1}{|B|}\sum_{(s,a,r,s',d) \in B} \left( Q_{\phi_i}(s,a) - y(r,s',d) \right)^2 && \text{for } i=1,2
                \end{align*}
                \IF{ $j \mod$ \texttt{policy\_delay} $ = 0$}
                    \STATE Update policy by one step of gradient ascent using
                    \begin{equation*}
                        \nabla_{\theta} \frac{1}{|B|}\sum_{s \in B}Q_{\phi_1}(s, \mu_{\theta}(s))
                    \end{equation*}
                    \STATE Update target networks with
                    \begin{align*}
                        \phi_{\text{targ},i} &\leftarrow \rho \phi_{\text{targ}, i} + (1-\rho) \phi_i && \text{for } i=1,2\\
                        \theta_{\text{targ}} &\leftarrow \rho \theta_{\text{targ}} + (1-\rho) \theta
                    \end{align*}
                \ENDIF
            \ENDFOR
            \ENDIF
        \UNTIL{convergence}
    \end{algorithmic}
    \end{algorithm}


Implementations
----------------
The default config is defined as follows:

.. autoclass:: ding.policy.td3.TD3Policy

Model
~~~~~~~~~~~~~~~~~
Here we provide examples of `td3` model as default model for `TD3`.

.. autoclass:: ding.model.template.qac.QAC
    :members: forward, compute_actor, compute_critic
    :noindex:

Train actor-critic model
~~~~~~~~~~~~~~~~~~~~~~~~~~
First, we initialize actor and critic optimizer in ``_init_learn``, respectively.
Setting up two separate optimizers can guarantee that we **only update** actor network parameters and not critic network when we compute actor loss, vice versa.

    .. code-block:: python

        # actor and critic optimizer
        self._optimizer_actor = Adam(
            self._model.actor.parameters(),
            lr=self._cfg.learn.learning_rate_actor,
            weight_decay=self._cfg.learn.weight_decay
        )
        self._optimizer_critic = Adam(
            self._model.critic.parameters(),
            lr=self._cfg.learn.learning_rate_critic,
            weight_decay=self._cfg.learn.weight_decay
        )

In ``_forward_learn`` we update actor-critic policy through computing critic loss, updating critic network, computing actor loss, and updating actor network.
    1. ``critic loss computation``

        - current and target value computation

        .. code-block:: python

            # current q value
            q_value = self._learn_model.forward(data, mode='compute_critic')['q_value']
            q_value_dict = {}
            if self._twin_critic:
                q_value_dict['q_value'] = q_value[0].mean()
                q_value_dict['q_value_twin'] = q_value[1].mean()
            else:
                q_value_dict['q_value'] = q_value.mean()
            # target q value. SARSA: first predict next action, then calculate next q value
            with torch.no_grad():
                next_action = self._target_model.forward(next_obs, mode='compute_actor')['action']
                next_data = {'obs': next_obs, 'action': next_action}
                target_q_value = self._target_model.forward(next_data, mode='compute_critic')['q_value']

        - target(**Clipped Double-Q Learning**) and loss computation

        .. code-block:: python

            if self._twin_critic:
                # TD3: two critic networks
                target_q_value = torch.min(target_q_value[0], target_q_value[1])  # find min one as target q value
                # network1
                td_data = v_1step_td_data(q_value[0], target_q_value, reward, data['done'], data['weight'])
                critic_loss, td_error_per_sample1 = v_1step_td_error(td_data, self._gamma)
                loss_dict['critic_loss'] = critic_loss
                # network2(twin network)
                td_data_twin = v_1step_td_data(q_value[1], target_q_value, reward, data['done'], data['weight'])
                critic_twin_loss, td_error_per_sample2 = v_1step_td_error(td_data_twin, self._gamma)
                loss_dict['critic_twin_loss'] = critic_twin_loss
                td_error_per_sample = (td_error_per_sample1 + td_error_per_sample2) / 2
            else:
                # DDPG: single critic network
                td_data = v_1step_td_data(q_value, target_q_value, reward, data['done'], data['weight'])
                critic_loss, td_error_per_sample = v_1step_td_error(td_data, self._gamma)
                loss_dict['critic_loss'] = critic_loss

    2. ``critic network update``

    .. code-block:: python

        self._optimizer_critic.zero_grad()
        for k in loss_dict:
            if 'critic' in k:
                loss_dict[k].backward()
        self._optimizer_critic.step()

    3. ``actor loss`` and  ``actor network update`` depending on the level of **delaying the policy updates**.

    .. code-block:: python

        if (self._forward_learn_cnt + 1) % self._actor_update_freq == 0:
            actor_data = self._learn_model.forward(data['obs'], mode='compute_actor')
            actor_data['obs'] = data['obs']
            if self._twin_critic:
                actor_loss = -self._learn_model.forward(actor_data, mode='compute_critic')['q_value'][0].mean()
            else:
                actor_loss = -self._learn_model.forward(actor_data, mode='compute_critic')['q_value'].mean()

            loss_dict['actor_loss'] = actor_loss
            # actor update
            self._optimizer_actor.zero_grad()
            actor_loss.backward()
            self._optimizer_actor.step()


Target Network
~~~~~~~~~~~~~~~~~
We implement Target Network trough target model initialization in ``_init_learn``.
We configure ``learn.target_theta`` to control the interpolation factor in averaging.


.. code-block:: python

    # main and target models
    self._target_model = copy.deepcopy(self._model)
    self._target_model = model_wrap(
        self._target_model,
        wrapper_name='target',
        update_type='momentum',
        update_kwargs={'theta': self._cfg.learn.target_theta}
    )


Target Policy Smoothing Regularization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We implement Target Policy Smoothing Regularization trough target model initialization in ``_init_learn``.
We configure ``learn.noise``, ``learn.noise_sigma``, and ``learn.noise_range`` to control the added noise, which is clipped to keep the target close to the original action.

.. code-block:: python

    if self._cfg.learn.noise:
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='action_noise',
            noise_type='gauss',
            noise_kwargs={
                'mu': 0.0,
                'sigma': self._cfg.learn.noise_sigma
            },
            noise_range=self._cfg.learn.noise_range
        )



Benchmark
-----------


+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
| environment         |best mean reward | evaluation results                                  | config link              | comparison           |
+=====================+=================+=====================================================+==========================+======================+
|                     |                 |                                                     |`config_link_ha <https:// |                      |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|                     |                 |                                                     |DI-engine/blob/main/dizoo/| Baselines(9535)      |
|Halfcheetah          |  12900          |.. image:: images/benchmark/halfcheetah_sac.png      |mujoco/config/halfcheetah_|                      |
|                     |                 |                                                     |sac_default_config.py>`_  | Tianshou(12138)      |
|(Halfcheetah-v3)     |                 |                                                     |                          |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_w <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|Walker2d             |                 |                                                     |DI-engine/blob/main/dizoo/| Baselines(3863)      |
|                     |  5172           |.. image:: images/benchmark/walker2d_sac.png         |mujoco/config/walker2d_   |                      |
|(Walker2d-v2)        |                 |                                                     |sac_default_config.py>`_  | Tianshou(5007)       |
|                     |                 |                                                     |                          |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_ho <https:// |                      |
|                     |                 |                                                     |github.com/opendilab/     | Baselines(2325)      |
|Hopper               |                 |                                                     |DI-engine/blob/main/dizoo/|                      |
|(Hopper-v2)          |  3653           |.. image:: images/benchmark/hopper_sac.png           |mujoco/config/hopper_sac_ | Tianshou(3542)       |
|                     |                 |                                                     |default_config.py>`_      |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+


Other Public Implementations
----------------------------

- `Official implementation`_

.. _`Official implementation`: https://github.com/sfujim/TD3_BC

References
-----------
Scott Fujimoto, Shixiang Shane Gu: “A Minimalist Approach to Offline Reinforcement Learning”, 2021; [https://arxiv.org/abs/2106.06860 arXiv:2106.06860].
