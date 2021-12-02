PixelCNN：一种基于密度模型计数的探索方法 
=========

概要
--------

基于计数的探索方法是强化学习中一种以状态访问次数作为新奇度度量并以此来设计内在奖励的方法。
在深度强化学习中用密度模型作基于技术的探索最早在论文 `Bellemare et al, 2016 <https://arxiv.org/abs/1606.01868>`__ 中被提出。在这篇论文里，作者定义了**伪计数**并使用密度模型来近似求解每个状态的访问次数。一种常见的基于计数的内在奖励为MBIE-EB: :math:`r_t^i = N(s_t,a_t)^{-1/2}`
(`Strehl & Littman, 2008 <https://pdf.sciencedirectassets.com/272574/1-s2.0-S0022000008X00078/1-s2.0-S0022000008000767/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEP7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIFL0MkuJ5BidfH21KQvuR0Ozim2M%2FordC9adn%2Fam9517AiEAmf7Y7NKFe3PG6BUaLLxmuw6X4%2BGRjI0b00irYMu%2FreUqgwQIt%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAEGgwwNTkwMDM1NDY4NjUiDAHql%2FSlOD0VnmqejCrXA1nKkGKFkcUvrKOhhFGNsJFs9dn2hUeqJmVmnwVHQe9aXaPZUHoa%2Ffi5Q6e5jzZANyRT4o9y%2BKrUrw3%2FgKgdHblmQXshowYtAyf5EBxqe4qqv69djo51vdzGuTCRBEDpD4w9cE%2Fw%2F%2BJtT6YyKpO3xFsllYEDaGjfADK7ZUlAre6TDYQzE4ZgdcfqVndoFudAeZBRoycNmmhdxblNUEZ09b5N4BJyvKLxcxxiL264cSOIQ0vY5rV6195Zl%2BX1INn48MKdF2jTBKNgD611EAwH6mmQuyN%2BCavLzjPlw6weov1bVDcJ6fbZKMJgrXbNHzptWDlceeHR83IwOrNAuSUjqY6kcsk0RDrupRJqtL9Z0lY%2BMYYyh4BICDoHgKdq%2FkAQP1DVFkUIKrrhVJHl1uqjznUTe2hurmudQAHO0QMnBtDS4CSVkv0KiLq8qikQe1shD5yMmEU8ASWZHOS8bjmki7NiR9G5lUQLcTeTIMmLXoH9SbXdNy5HKLOTHAp5KzRTMxZSmIo5M%2BuZYh3gNQhtm%2Bcasybr%2BNs0XqZ3ba2Sl1x91agbY1HZul9lUz%2FLT%2BZnWz9%2BNRgzDhBIWvOnN0Jtli%2BV%2BdHLayQdxgIczkZN%2FtmAYADEAYC%2FbDD%2F1NeMBjqlASheTXg0fWjiG5ilejoJADr9VL07uLY5dlTEgBU3UyWXyZgwLCkZd5E6RN3LUl%2BfYjd0CZcnq9y2K6%2FIB43%2F4l8ZjFjkoYXFY971RdCVwXc%2B0XEs%2Fs%2BkZ3fS1okNyhZSTa5vuGtpKCwRKw0z4izi0UmNgIecb%2BCA1e5plIx0phAFcozqxS4sCXNvDA2Ez8VRCCDJ7z0jJy%2BBGhWrhuIY9JPUYnUdyw%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20211118T065530Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYZZML3HPC%2F20211118%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=e22022d996c065bf378b18532a76779fc40ae742cede61d54ab03e1d67dc20ff&hash=3b0878c83992eb605cbc57f63e35e7b3eab6b859dd5eaadc1cf2211f47fbba17&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0022000008000767&tid=spdf-96e5fc2d-31ad-47d8-8ca6-813bb30d7716&sid=950cbe9b17b4e247ab88f2848953ee673386gxrqa&type=client>`__)，
基于伪计数的探索奖励与它形式相似：

.. math::
    r_t^i = (\hat{N}_n(s_t,a_t) + 0.01)^{-1/2}

在 `Bellemare et al, 2016 <https://arxiv.org/abs/1606.01868>`__ 中，密度模型是一个简单的上下文相关树(CTS)，其作用为计算状态某一像素在其左上像素
邻居的条件下的计数密度。上下文相关树模型简单，但存在一定局限性如：表达能力不强，不能拓展到大规模问题和数据效率不高等。因此，在接下来的一篇论文中
`Goerg Ostrovski et al, 2017 <https://arxiv.org/abs/1703.01310>`__ 改进了CTS，使用表达能力更强的PixelCNN(`van den Oord et al, 2016 <https://arxiv.org/abs/1606.05328>`__)
来作为密度模型。

快速上手
-----------

1. 在强化学习中探索方法的原理是：我们首先建立一种测量状态的“新颖性”的方法，然后我们分配一个与状态的新颖性成比例的探索奖励。
 
2. PixelCNN是一种基于**伪计数**的探索方法，可以应用于非表格的情况。基于计数的方法的主要思想是通过计算状态访问次数来定义内在奖励。

3. 在基于PixelCNN计数的论文中，使用的基本强化学习算法是DQN，但是基于这种方法给予的内在奖励可以用于任何强化学习算法中。

关键公式与图片
---------------------------

构造密度模型 :math:`p_{\theta}(s)` (或者 :math:`p_{\theta}(s,a)`) 是基于计数的探索方法的关键。
然而在深度强化学习背景下，即使新状态 :math:`s` 与之前见过的状态很相似，:math:`p_{\theta}(s)` 的值仍然可能很高。
为此，我们利用密度概率真值与状态访问次数的关系来定义伪计数。

密度概率真值与状态访问次数的关系为：

.. math::
   P(s) &= \frac{N(s)}{n}\\
   P'(s) &= \frac{N(s)+1}{n+1}

在小规模，表格型MDP中，:math:`n` 和 :math:`N(s)` 都是可以计算的，但在深度强化学习的背景下，它们有时不可测量或计算代价很高。因此，我们引入伪计数。

基于伪计数的算法定义如下：

    loop:
        1.fit model :math:`p_{\theta}(s)` to all states :math:`D` seen so far

        2.take a step i and observes :math:`s_i`
        
        3.fit new model :math:`p_{\theta{'}}(s)` to :math:`{D \cup {s_i}}`
        
        4.use :math:`p_{\theta}(s_i)` and :math:`p_{\theta{'}}(s_i)` to estimate :math:`\hat{N}(s)`
        
        5.set :math:`{r_i^{+} = r_i + B(\hat{N}(s))}`

其中，:math:`{\hat{N}(s)}` 是伪计数。但是问题来了？我们如何得到伪计数呢？观察上面的式子和算法流程，我们有两个方程和两个未知数，可以同过密度模型的预测概率直接解出伪计数！

.. math::
    \hat{N}(s_i) &= \hat{n}p_{\theta}(s_i)\\
    \hat{n} &= \frac{1-p_{\theta{'}}(s_i)}{p_{\theta{'}}(s_i)- p_{\theta}(s_i)}p_{\theta}(s_i)




为了理解更多内在奖励的细节，推荐阅读 `原论文 <https://arxiv.org/abs/1703.01310>`__。

实现
---------------

PixelCNN奖励模型的接口定义如下：

.. autoclass:: ding.reward_model.count_based_model.CountbasedRewardModel
   :members: __init__, estimate
   :noindex:

PixelCNN奖励模型的实现如下：

.. autoclass::  ding.torch_utils.network.GatedPixelCNN
   :members: __init__, forward
   :noindex:

DQN算法实现如下：

.. autoclass:: ding.policy.dqn.DQNPolicy
   :noindex:

完整代码，请参考我们DI-engine的 `实现 <https://github.com/opendilab/DI-engine/blob/main/ding/reward_model/count_based_model.py>`__


基准结果
----------------------------

minigrid-empty-8x8:


minigrid-fourrooms-8x8:

atari-Gravitar:

atari-Pitfall:



作者的官方Tensorflow实现
----------------------------

- PixelCNN_

.. _PixelCNN: https://github.com/openai/pixel-cnn.


参考资料
---------

1. Bellemare, Marc, et al. "Unifying count-based exploration and intrinsic motivation." Advances in neural information processing systems 29 (2016): 1471-1479.

2. Strehl, Alexander L., and Michael L. Littman. "An analysis of model-based interval estimation for Markov decision processes." Journal of Computer and System Sciences 74.8 (2008): 1309-1331.

3. Ostrovski, Georg, et al. "Count-based exploration with neural density models." International conference on machine learning. PMLR, 2017.

4. Oord, Aaron van den, et al. "Conditional image generation with pixelcnn decoders." arXiv preprint arXiv:1606.05328 (2016).

5. https://lilianweng.github.io/lil-log/2020/06/07/exploration-strategies-in-deep-reinforcement-learning.html#count-based-exploration

6. http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-13.pdf
