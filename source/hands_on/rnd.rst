RND
====

Overview
--------

RND (Random Network Distillation) was first proposed in
`Exploration by Random Network Distillation <https://arxiv.org/abs/1810.12894v1>`__, which introduce
an exploration bonus for deep reinforcement learning methods that is easy to implement and adds minimal
overhead to the computation performed. The bonus is the error of a neural network predicting features
of the observations given by a fixed randomly initialized neural network.

Quick Facts
-----------

1. RND is a prediction-error-based exploration approach that can be applied in non-tabular cases.
   The key problem behind exploration approaches is that how to measure the novelty of states, if the state
   is more novel, it will matching a bigger intrinsic reward, vice versa.
   The insight of prediction-error-based approaches is that defining the intrinsic reward as the prediction error
   for a problem related to the agent’s transitions, such as learning forward dynamics model, learning
   inverse dynamics model, or even a randomly generated problem, which is the case in RND algorithm.

2. RND involves two neural networks: a fixed and randomly initialized target network which sets the prediction problem,
    and a predictor network trained on data collected by the agent. The target network takes an observation to an embedding
   :math:`f: O → R^k` and the predictor neural network :math:`\hat{f}: O → R^k` is trained by gradient descent to minimize
   the expected MSE :math:`∥f (x; θ) − f (x)∥` with respect to its parameters :math:`θ_\hat{f}`.
:math:``


3. The RND intrinsic reward generation model can be combined with different RL algorithms conveniently.

Key Equations or Key Graphs
---------------------------

The overall sketch of RND is as below:

.. figure:: images/RND.png
   :align: center
   :scale: 85%
   :alt:

In RND paper points out that prediction errors can be attributed to the following 4 factors:

1. Amount of training data. Prediction error is high where few similar examples were seen by the predictor
(epistemic uncertainty).

2. Stochasticity. Prediction error is high because the target function is stochastic (aleatoric un- certainty).
Stochastic transitions are a source of such error for forward dynamics prediction.

3. Model misspecification. Prediction error is high because necessary information is missing,
or the model class is too limited to fit the complexity of the target function.

4. Learning dynamics. Prediction error is high because the optimization process fails to find a
predictor in the model class that best approximates the target function.

RND obviates factors 2 and 3 since the target network can be chosen to be deterministic and has the identical network structure with
the model-class of the predictor network. But RND is still facing a problem that the rnd bonus reward is gradually disappears over time.


Pseudo-Code
-----------

.. figure:: images/RND_pseudo_code.png
   :alt:

Implementation
---------------
The default config of on policy PPO is defined as follows:

.. autoclass:: ding.policy.ppo.PPOPolicy
   :noindex:

The RND reward model is defined as follows:

.. autoclass:: ding.reward_model.rnd_reward_model.RndRewardModel
   :members: __init__
   :noindex:

Train RND reward model
~~~~~~~~~~~~~~~~~~~~~~~~~

First, we initialize reward model and optimizer in ``_init_`` of class ``RndRewardModel``.

        .. code-block:: python

            self.reward_model = RndNetwork(config.obs_shape, config.hidden_size_list)

            self.intrinsic_reward_type = config.intrinsic_reward_type
            assert self.intrinsic_reward_type in ['add', 'new', 'assign']
            self.train_data = []
            self.opt = optim.Adam(self.reward_model.predictor.parameters(), config.learning_rate)
            self._running_mean_std_rnd = RunningMeanStd(epsilon=1e-4)



Then, we calculate the reward model loss and update the RND reward model network.

        .. code-block:: python

         def _train(self) -> None:
            train_data: list = random.sample(self.train_data, self.cfg.batch_size)
            train_data: torch.Tensor = torch.stack(train_data).to(self.device)
            predict_feature, target_feature = self.reward_model(train_data)
            loss = F.mse_loss(predict_feature, target_feature.detach())
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()


Finally, we rewrite the reward key in each row of the data in ``estimate`` of class ``RndRewardModel``.


    1. ``calculate the rnd pseudo reward``

        .. code-block:: python

            obs = collect_states(data)
            obs = torch.stack(obs).to(self.device)
            with torch.no_grad():
                predict_feature, target_feature = self.reward_model(obs)
                reward = F.mse_loss(predict_feature, target_feature, reduction='none').mean(dim=1)
                self._running_mean_std_rnd.update(reward.cpu().numpy())
                reward = reward / self._running_mean_std_rnd.std

    2. ``combine the rnd pseudo reward with the original reward``

        .. code-block:: python

            for item, rew in zip(data, reward):
            if self.intrinsic_reward_type == 'add':
                item['reward'] += rew
            elif self.intrinsic_reward_type == 'new':
                item['intrinsic_reward'] = rew
            elif self.intrinsic_reward_type == 'assign':
                item['reward'] = rew

The author's tensorflow Implementations
----------------------------

- RND_

.. _RND: https://github.com/openai/random-network-distillation.

Reference
---------

1. Burda Y, Edwards H, Storkey A, et al. Exploration by random network distillation[J]. arXiv preprint arXiv:1810.12894, 2018.

2. https://openai.com/blog/reinforcement-learning-with-prediction-based-rewards/

3. https://lilianweng.github.io/lil-log/2020/06/07/exploration-strategies-in-deep-reinforcement-learning.html#prediction-based-exploration
