Reward Model 使用指南
===============================

Reward Model 入门
-------------------------------

**Reward Model 的基本概念**

在强化学习中，Reward Model 是对智能体的行为进行评价的模型，它的输入是智能体的观测和动作，输出是一个标量的奖励值。

在 DI-engine 中，Reward Model是我们提供的一个组件。所有的Reward Model类都继承自名为\ **BaseRewardModel**\的一个抽象基类。在这个类中，我们定义了最基本的Reward Model功能如下。

.. code-block:: python
    
   class BaseRewardModel(ABC):

        @abstractmethod
        def estimate(self, data: list) -> Any:
            """
            给出估计的奖励值
            """

        @abstractmethod
        def train(self, data) -> None:
            """
            训练Reward Model
            """

        @abstractmethod
        def collect_data(self, data) -> None:
            """
            收集RM所需要的训练数据
            """

        @abstractmethod
        def clear_data(self, iter: int) -> None:
            """
            清理Reward Model类buffer里的数(即collect_data收集的数据)
            """

        def load_expert_data(self, data) -> None:
            """
            加载专家数据，只有在使用专家数据训练Reward Model时才需要实现
            """

**快速使用Reward Model**

在 DI-engine 中，我们提供了一些常用的Reward Model，用户可以直接使用。这些Reward Model都被注册到了\ **registry**\中，用户可以通过调用\**serial_pipeline_reward_model_offpolicy**\或者\**serial_pipeline_reward_model_onpolicy**\快速将reward model添加到自己的强化学习训练中。


.. code-block:: python

    from ding.entry import serial_pipeline_reward_model_offpolicy

    # 你所要训练的main config， create config
    # cooptrain_reward = True 表示在训练policy的时候，同时训练reward model
    # pretrain_reward = False 表示在开始训练前，不预训练reward model
    # 需要注意的是，如果想要使用pretrain_reward，需要在init reward model的时候准备好train data
    # 目前只有trex支持pretrain_reward
    serial_pipeline_reward_model_offpolicy((main_config, create_config), cooptrain_reward = True, pretrain_reward = False)



**在强化学习训练中添加Reward Model**

在上一小节中，我们介绍了 Reward Model在DI-engine中提供的外部方法。接下来，我们将介绍如何在强化学习训练中添加Reward Model。

.. code-block:: python
    
    from ding.reward_model import create_reward_model

    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    # 创建Reward Model
    # 可以被创建的Reward MOdel可以从ding.reward_model中查看
    # 只可以创建被registry注册过的Reward Model
    reward_model = create_reward_model(cfg.reward_model, policy.collect_mode.get_attribute('device'), tb_logger)

**训练Reward Model**

在使用Reward Model预测奖励前，我们往往需要先训练Reward Model。接下来，我们将展示如何为reward model收集数据并训练

.. code-block:: python
    
    # 用DI-engine中的Collector类收集数据
    new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
    # 将collector收集到的数据传入reward model中，存储在reward model的buffer或者list中
    reward_model.collect_data(new_data)
    # 训练reward model
    reward_model.train()
    # 清理reward model的buffer或者list，
    # 传入当前训练的iteration，清理的间隔是受reward model config文件中的clear_buffer_per_iters参数控制的
    reward_model.clear_data(iter=learner.train_iter)

**用Reward Model预测reward**

接下来，我们将展示，如何用reward model预测reward， 并将预测好的reward加入到我们RL算法的训练过程当中。

.. code-block:: python
    
    # 获取训练agent所需要的数据，其中至少要包括 obs, action, reward, next_obs, done
    train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
    # 用Reward Model预测reward
    # estimate函数会返回一个 list[Dict],其中每个Dict都至少包含obs, action, reward, next_obs, done
    # estimate会修改传入的train_data的reward为Reward Model预测的reward
    train_data_augmented = reward_model.estimate(train_data)
    # 将预测好的reward加入到我们RL算法的训练过程当中
    learner.train(train_data_augmented, collector.envstep)

如何添加新的Reward Model
-------------------------------

在上一节中，我们介绍了如何使用现有的reward model。接下来，我们将展示如何添加新的reward model，并且需要遵循哪些规范。


**简介**

目前，DI-engine中的RM主要由两个类构成。 一个是继承自BaseRewardModel的RewardModel类，这个类主要提供四个公开方法，分别是collect_data, clear_data, train 和 estimate. 另一个是继承自torch.nn.Module的Reward Model Network类， 这个类会在RewardModel的初始化函数中初始化，主要包括forward和learn两个公开方法。
然后，统一的RM都会由统一的两个entry调用(onpolicy or offpolicy)。

**RewardModel方法介绍**

首先你的reward model需要包含以下方法，具体方法的介绍如下。

.. code-block:: python
    
    def __init__(self, config: EasyDict, device: str, tb_logger: 'SummaryWriter') -> None: 
    """
    初始化RM，会在create RM的时候调用，需要注意的是如果要导入expert data（可以写一个self.load_expert_data()的方法）
    请在这里导入（pretrain目前必须导入expert data）
    初始化包含的内容有，reward_model的config, reward model network, logger 和counter
    self.cfg: 是config中的config.reward_model
    self.reward_model: 我们自己写的Reward model Network
    self.tb_logger: 目前使用的是tb_logger
    self.counter: 各种你需要的计数器
    """
    pass

    def collect_data(self, data: list) -> None:
    """
    这个方法的在entry的inner loop中调用
    这个方法的用途是在coop-train的时候，向RM增加新的数据（不用于pretrain）
    传入的data应该是一个由dict组成的list， 
    每个dict需要包含（特殊情况请在注释中写明，推荐用assert确定在运算前）
    {"obs": torch.tensor, "next_obs": torch.tensor, "action": torch.tensor, "reward": torch.tensor}
    example:
    self.train_data.extend(data)
    """
    pass

    def clear_data(self, iter: int) -> None:
    """
    这个方法的在entry的inner loop中调用，作用是定期清除RM所用的train data
    注意）：不是所有的RM都需要clear_data
    传入的参数是当前train的iteration
    example:
    assert hasattr(self.cfg, 'clear_buffer_per_iters'), "Reward Model does not have clear_buffer_per_iters, Clear failed"
    if iter % self.cfg.clear_buffer_per_iters == 0:
        self.train_data.clear()
    """
    pass

    def train(self) -> None:
    """
    这个方法的在entry的inner loop中调用，
    功能是train整个RM，并向logger添加内容，形式应该如下
    1. 由内部方法_train()进行具体训练，接受需要加入logger的返回值
    2. 将对应内容添加到logger
    for _ in range(self.cfg.update_per_collect):
        loss = self._train()
        self.tb_logger.add_scalar('reward_model/reward_loss', loss, self.train_iter)
        self.train_iter += 1
    """
    pass

    def estimate(self, data: list) -> List[Dict]:
    """
    这个方法在entry的inner loop中调用
    输入收集来的data，将data中的item['reward']替换为RM提供的reward，因此输入和输出在形式上应该保持统一
    example:
         with torch.no_grad():
            reward = self.reward_model.forward(res).squeeze(-1).cpu()
         for item, rew in zip(train_data_augmented, reward):
            item['reward'] = -torch.log(rew + 1e-8)
    """
    pass

**RMNetwork方法介绍**

你需要在RewardModel初始化的时候初始化RMNetwork，并且你的RMNetwork需要包含以下方法，具体方法的介绍如下。

.. code-block:: python

    def forward(self, data: torch.Tensor) -> torch.Tensor:
    """
    用于返回RM Network给出的reward，返回值是reward
    data不需要严格遵守，可以根据具体情况选择输入
    example:
         reward = self.feature(data)
        return reward
    """
    pass

    def learn(self, data: torch.Tensor) -> torch.Tensor:
    """
    用于在RM._train()中调用，返回的是loss
    data不需要严格遵守，可以根据具体情况选择输入
    example：
        loss = xxx
        return loss
    """
    pass


**(补充)**

- 1. 所有的RM必须提供entry测试，现有的测试在 `here <https://github.com/opendilab/DI-engine/blob/main/ding/entry/tests/test_serial_entry_reward_model.py>`_

- 2. 所有RM的简单运行环境config都在 `here <https://github.com/opendilab/DI-engine/tree/main/dizoo/classic_control/cartpole>`_