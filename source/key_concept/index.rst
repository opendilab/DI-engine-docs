Key Concept
===============================

.. toctree::
   :maxdepth: 3


在Key Concept部分，我们主要介绍DI-engine中有关强化学习训练和评估的一些关键概念。
其中一个基本的控制模式（串行模式）的运行逻辑图如下：

.. image::
   images/serial_pipeline.svg
   :align: center

在下面的章节中，DI-engine首先介绍了强化学习的关键的概念和重要的相关组件，然后将它们结合起来构成运行逻辑图，以提供不同的计算模式。(计算模式包括serial模式, parallel模式, dist模式).

概念
----------
``Environment（环境）``和 ``policy（策略）``是DI-engine设计方案中最重要的两个概念，在大多数情况下，DI-engine的用户只需要关注这两个部分。

``Worker（工作者）``模块，如 ``learner（学习者）``, ``collector（收集者）``和 ``buffer（缓冲器）``，是用来执行具体任务的执行模块。这些工作模块在许多RL算法中是通用的，但是用户也可以自己定制相应模块，自己定制模块时需要遵守相应的接口定义。

``config（配置文件）``是我们推荐的用于控制和记录整个训练过程的模块。

.. tip::
  我们的环境和策略在其他RL算法的原始定义上进行了部分扩展。

环境
~~~~~~~~~~~~~
DI-engine环境是``gym.Env``的超集，包含了gym.Env，它与gym env接口兼容，并提供一些可选的接口。例如: dynamic seed, collect/evaluate setting, `具体可查看 <../feature/env_overview_en.html>`_

``EnvManager``, 通常在其他框架中被称为 "矢量环境"，用于实现并行的环境模拟，以加快数据收集速度。它允许采集器在每个采集步骤中与N个同类型环境交互，而不是只与1个环境交互，这意味着传递给`env.step'的`action'是一个长度为N的向量，`env.step'的返回值（obs, reward, done）与只与一个环境交互时相同。

为了方便异步复位和统一异步/同步操作时的相应接口，DI-engine修改了env manager的接口，具体情况如下：

.. code:: python

   # DI-engine EnvManager                                                         # pseudo code in the other RL papers
   env.launch()                                                                   # obs = env.reset()
   while True:                                                                    # while True:
       obs = env.ready_obs                                                              
       action = random_policy.forward(obs)                                        #     action = random_policy.forward(obs)
       timestep = env.step(action)                                                #     obs_, reward, done, info = env.step(action)
       # maybe some env_id matching when enable asynchronous
       transition = [obs, action, timestep.obs, timestep.reward, timestep.done]   #     transition = [obs, action, obs_, reward, done]
                                                                                  #     if done:
                                                                                  #         obs[i] = env.reset(i)
       if env.done:                                                               #     if env.done  # collect enough env frames
           break                                                                  #         break

目前DI-engine中有三种类型的EnvManager：

  - BaseEnvManager——用于本地测试和验证
  - SyncSubprocessEnvManager——用于低波动环境的并行模拟
  - AsyncSubprocessEnvManager——用于高波动环境的并行模拟

下面的演示图显示了 "BaseEnvManager "和 "SyncSubprocessEnvManager "的详细运行逻辑。

.. image::
   images/env_manager_base_sync.png

对于子进程类型的环境管理器，目前DI-engine在不同的工作子进程之间使用共享内存，以节省IPC的成本．也可以参考`pyarrow <https://github.com/apache/arrow>`_，它在接下来的版本中会成为一个可靠的新的替代方案。

.. note::
   如果环境需要启动某种客户端，像SC2和CARLA，有些时候基于python线程的新环境管理器可以更快。

.. note::
   如果在使用GPU的环境中存在一些预定义的神经网络，比如在RL训练前先通过自监督训练的特征提取器VAE．目前的DI-engine建议在每个子进程中并行执行这一特征提取器，而不是在主进程中堆叠所有数据，然后再转发特征提取器供其使用。DI-engine还将试图找到更好的方法来解决这一问题。

此外，为了保证实际使用中的鲁棒性，如解决遇到IPC错误（断管、EOF）和环境运行错误的问题，DI-engine提供了一系列的容错工具，例如：autodog(看门狗)和auto-retry(自动重试)。

对于所有提到的功能，用户可以参考 "EnvManager概述<.../feature/env_manager_overview_en.html>"了解更多细节。

策略
~~~~~~~
为了统一RL和其他机器学习算法的设计模式并将其模块化，DI-engine抽象并定义了具有多模式设计的通用策略接口。
有了这些接口，大量的人工智能决策算法可以只在一个python文件中实现，仅需实现对应的策略类。用户的定制算法只需要继承和扩展:class:`Policy <ding.policy.Policy>`，或者只需要与它有相同的接口定义即可。

多模式策略
^^^^^^^^^^^^^^^^^^^^^^^^^^
在大多数情况下，RL策略需要为不同的用途执行不同的算法程序，例如：对于DQN算法，在训练中需要执行模型前向传播和计算TD误差。模型需要在没有梯度计算的情况下进行前向传播，并使用epsilon-greedy来选择行动进行探索和收集。DI-engine将算法策略的不同内容统一在了一个python文件中。
我们准备了一些简单的接口方法，并将它们组合成3种常用模式--**学习模式、收集模式、评估模式**，如下图所示。

.. image::
   images/policy_mode.svg

Learn_mode旨在进行策略更新，Collect_mode进行适当的探索和收集训练数据，Eval_mode用于评估策略的性能。用户可以通过覆盖这些模式来定制自己的算法，或者设计自己的定制模式，如根据训练结果进行超参数退火，或在自主游戏训练中选择战斗队员等。了解更多细节,
用户可以参考 "策略概述<.../feature/policy_overview_en.html>"_。

.. note::
   `policy.learn_mode`不是:class:`Policy <ding.policy.Policy>`的实例，而是一个纯接口集合（由namedtuple实现），这意味着用户可以实现他们的政策类，只需确保与相应的模式有相同的方法名称和输入/输出参数。

共享模型+模型封装器
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
神经网络，通常称为模型，是整个算法中最重要的组成部分之一。对于串行模式，模型通常在公共的通用构造方法（``__init__``）中创建，或者在策略中创建，将其参数传递给模型。为了方便起见，模型在不同的模式中是被共享的。DI-engine通过 "模型封装器（model_wrapper) "为模型扩展了更多功能，这使得共享的模型在不同的模式下可以表现出不同的行为，例如在收集模式下通过分布进行采样，而在评估模式下则用于评估策略好坏。下面是一些具体的代码例子。

.. code:: python

    from ding.model import model_wrap, DQN

    model = DQN(obs_shape=(4, 84, 84), action_shape=6)
    # only wrapper, no model copy
    learn_model = model_wrap(model_wrap, wrapper_name='base')
    collector_model = model_wrap(model, wrapper_name='multinomial_sample')
    eval_model = model_wrap(model, wrapper_name='argmax_sample')

如果你想了解更多关于每种算法的预定义模型或者定制你的模型，请参考`模型概述<.../feature/model_overview_en.html>`_。

如果你想了解预定义模型包装器的详细信息或定制你的模型包装器，请参考`包装器概述<.../feature/wrapper_hook_overview_en.html>`。

数据处理函数
^^^^^^^^^^^^^^^^^^^^^^
在实际的算法实现中，用户往往需要很多数据处理操作，比如将几个样本堆叠成一个批次，在torch.Tensor和np.ndarray之间进行数据转换。至于RL算法本身，有大量不同风格的数据预处理和聚合，如计算N步返回和GAE（Generalized Advantage Estimation），分割轨迹或解卷段，等等。DI-engine提供了一些常用的处理函数，这些函数可以作为一个纯函数来调用，用户可以在收集模式和学习模式中使用这些功能。

对于一些算法策略，如A2C/PPO，我们应该在学习模式还是收集模式中计算advantages？前者可以在分布式训练中把计算分配给不同的收集器节点，以节省时间，而后者由于更精确的近似，通常可以获得更好的性能。对于我们的框架来说，我们提供了一些强大而高效的计算工具，而不是将他们限制在了一个具体的地方只允许其在限制的地方调用。用户可以根据自己的需求选择在哪里调用这些函数。下表显示了一些现有的处理函数和它们的相关信息。


====================== ========================================== ==============================
Function Name          Description                                Path
====================== ========================================== ==============================
default_collate        Stack samples(dict/list/tensor) into batch ding.utils.data.collate_fn
default_decollate      Split batch into samples                   ding.utils.data.collate_fn
get_nstep_return_data  Get nstep data(reward, next_obs, done)     ding.rl_utils.adder
get_gae                Get GAE advantage                          ding.rl_utils.adder
to_tensor              Transform data to torch.Tensor             ding.torch_utils.data_helper
to_device              Transform device(cpu or cuda)              ding.torch_utils.data_helper
====================== ========================================== ==============================

扩大到并行训练
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TBD

配置文件
~~~~~~~~~

关键概念
^^^^^^^^^^^^

配置模块是一个供用户确定他们要使用哪些参数的组件。
整体设计如下。

.. image:: images/config.png
   :alt: 

正如你在上图中看到的，整个配置主要由两部分组成。一个叫做Default Config，这是我们推荐的默认参数，一般不会有太大的变化。另一个叫做 User Config，用户可以逐一指定具体参数。

为了获得整体的配置参数，我们有编译阶段，这是一个自下而上的过程。首先，我们得到每个子模块的默认配置参数，如学习者、收集器等模块的参数，接着得到策略和环境的默认参数。最后，我们将默认参数与用户配置参数合并，得到整个配置参数。

另一方面，在初始化阶段，即根据配置参数创建模块的过程，是自上而下的。我们将从策略和环境开始，然后将配置传递给每个执行模块。

在DI-engine中，我们把config写成一个python的`dict'`。下面是一个默认配置的config。

.. code:: python

   cartpole_dqn_default_config = dict(
       env=dict(
           manager=dict(...),
           ...
       ),
       policy=dict(
           model=dict(...),
           collect=dict(...),
           learn=dict(...),
           eval=dict(...),
           other=dict(
               replay_buffer=dict(),
               ...
           ),
           ...
       ),
   )

配置文件概述
^^^^^^^^^^^^^^^^

在下表中，我们列出了一些常用的关键参数以及它们的含义。
与策略有关的参数，请参考文件`Hands On　RL <.../hands_on/index.html>`__部分。

+-------------------------------+-------------------------------------+
| Key                           | Meaning                             |
+===============================+=====================================+
| policy.batch_size             | (int) number of data for a training |
|                               | iteration                           |
+-------------------------------+-------------------------------------+
| policy.update_per_collect     | (int) after getting training data,  |
|                               | leaner will update model for        |
|                               | update_per_collect times            |
+-------------------------------+-------------------------------------+
| policy.n_sample               | (int) number of samples that will   |
|                               | be sent to replay_buffer from       |
|                               | collector                           |
+-------------------------------+-------------------------------------+
| policy.nstep                  | (int) number of steps that will be  |
|                               | used when calculating TD-error.     |
+-------------------------------+-------------------------------------+
| policy.cuda                   | (bool) whether to use cuda when     |
|                               | training                            |
+-------------------------------+-------------------------------------+
| policy.priority               | (bool) whether to use priority      |
|                               | replay buffer                       |
+-------------------------------+-------------------------------------+
| policy.on_policy              | (bool) whether to use on policy     |
|                               | training                            |
+-------------------------------+-------------------------------------+
| env.stop_value                | (int) when reward exceeds           |
|                               | env.stop_value, stop training       |
+-------------------------------+-------------------------------------+
| env.collector_env_num         | (int) number of environments to     |
|                               | collect data when training          |
+-------------------------------+-------------------------------------+
| env.evaluator_env_num         | (int) number of environments to     |
|                               | collect data when evaluating        |
+-------------------------------+-------------------------------------+

合并用户特定配置和预定义配置时的规则：

-  用户特定的配置是最优先的，这意味着当冲突发生时，它将覆盖默认的配置。

-  一些重要的参数，如`env.stop_value', `env.n_evaluator_episode', `policy.on_policy`,　`policy.collect.n_sample'或`policy.collect.n_episode'必须是特定的。

-  合并后的整个配置将被保存到``total_config.py``和``formatted_total_config.py``中。

.. _header-n125:

如何定制？
^^^^^^^^^^^^^^^^^^

假设我们需要将上面提到的键`nstep`设置为3，如何做？

如果用户配置的文件名是``dqn_user_config.py``，只要在用户配置中加入以下代码：

.. code:: python

   policy=dict(
       ...,
       learn=dict(
           ...,
           nstep=3,
       )
   )

在写完用户配置后，我们可以参考 "快速启动<.../quick_start/index.html>"运行我们的DQN实验。

Worker-收集器
~~~~~~~~~~~~~~~~~~
收集器（collector）是所有Worker组件中最重要的组成部分之一，它在其他框架中通常被称为 "行为者"，DI-engine将其重新命名，以区别于行为者-批评者。它的目的是为学习者提供足够数量和高质量的数据。收集器只负责收集数据，与数据管理解耦，也就是说，它直接返回收集的数据，这些数据可以直接用于训练或存入Replay Buffer(缓冲区)。

收集器有3个核心部分--环境管理器、策略（collect_mode）、收集器控制器，这些部分可以在一个进程中实现，也可以位于几个机器中。通常，DI-engine使用一个多进程的env_manager和另一个存有策略的主循环控制器进程来构建一个收集器，这在将来可能被扩展。

由于发送和接收数据的逻辑不同，现在的收集器分为两种模式--串行和并行，我们将分别介绍它们。

串行收集器
^^^^^^^^^^^^^^^^^^^
从收集数据的基本单位来看，样本sample和剧集episode是两种主要使用的类型。因此，DI-engine为串行采集器定义了抽象接口`ISerialCollector'，并实现了`SampleCollector'和`EpisodeCollector'，这几乎涵盖了RL的使用，但当遇到一些特殊需求时，用户也可以进行定制。

.. image::
   images/serial_collector_class.svg
   :align: center


收集器的核心用法很简单，用户只需要创建一个相应的收集器类型，并将`n_sample'或`n_episode'作为`collect'方法的参数。下面是一个简单的例子。


.. code:: python

    import gym
    from easydict import EasyDict
    from ding.policy import DQNPolicy
    from ding.env import BaseEnvManager
    from ding.worker import SampleCollector, EpisodeCollector

    # prepare components
    cfg: EasyDict  # config after `compile_config`
    normal_env = BaseEnvManager(...)  # normal vectorized env
    
    dqn_policy = DQNPolicy(cfg.policy)
    sample_collector = SampleCollector(cfg.policy.collect.collector, normal_env, dqn_policy.collect_mode)
    episode_collector = EpisodeCollector(cfg.policy.collect.collector, normal_env, dqn_policy.collect_mode)

    # collect 100 train sample
    data = sample_collector.collect(n_sample=100)
    assert isinstance(data, list) and len(data) == 100
    assert all([isinstance(item, dict) for item in data])

    # collect 10 env episode
    episodes = episode_collector.collect(n_episode=10)
    assert isinstance(episodes, list) and len(episodes) == 10

    # push into replay buffer/send to learner/data preprocessing

.. note::
    对于所有的情况，收集数据的数量n_sample/n_episode，在整个训练过程中是固定的，我们的示例代码在配置中设置了这个字段，例如``config.policy.collect.n_sample``。


收集器的结构和主循环方式可以总结为下图，策略和环境的交互由`policy.forward', `env.step'和相关的支持代码组成。然后`policy.process_transition'和`policy.get_train_sample'用于将数据处理成训练样本并打包成一个List。对于`EpisodeCollector'中数据后处理的情况，除了`policy.get_train_sample'被禁用外，用户在收集到数据后可以做其它任何事情。

.. image::
   images/collector_pipeline.svg
   :align: center

有时，我们使用不同的策略甚至不同的环境来收集数据，比如在训练开始时使用随机策略来准备初始数据，用专家策略的概率来计算损失函数。所有的需求都可以通过 "reset_policy"、"reset_env"、"reset "这样的方法来实现。

.. code:: python
   
    # prepare components
    dqn_policy = DQNPolicy(...)
    random_policy = RandomPolicy(...)
    expert_policy = ExpertPolicy(...)

    collector = SampleCollector(...)
    replay_buffer = NaiveBuffer(...)

    # train beginning(random_policy)
    collector.reset_policy(random_policy.collect_mode)
    random_data = collector.collect(n_sample=10000)
    replay_buffer.push(random_data)
    # main loop
    while True:
        ...
        collector.reset_policy(dqn_policy.collect_mode)
        data = collector.collect(n_sample=100)
        collector.reset_policy(expert_policy.collect_mode)
        expert_data = collector.collect(n_sample=100)
        # train dqn_policy with collected data
        ...


此外，串行收集器在on-policy和off-policy算法之间差异较少，唯一的事情是在on-policy模式下需要设置buffer大小等统计参数，随后可以由收集器自动执行，用户只需确保`config.policy.on_policy`的正确值即可。

最后，还有一些其他功能，如用异步env_manager收集数据，处理异常的env步骤，请参考`收集器概述<.../feature/collector_overview_en.html>`。

并行收集器
^^^^^^^^^^^^^^^^^^^
TBD

Worker-缓冲器
~~~~~~~~~~~~~~~

缓冲器是一个组件，它用于存储由收集器收集的数据或由固定策略（通常是专家策略）产生的数据，然后为学习者提供数据以优化策略。在DI-engine中，有３种类型的缓冲器。

   - NaiveReplayBuffer
   - AdvancedReplayBuffer
   - EpisodeReplayBuffer

这三个都是源自抽象接口``IBuffer``的子类。

.. image::
   images/buffer_class_uml.png
   :align: center
   :scale: 50%

缓冲器的关键方法是``push'和``sample'。`NaiveReplayBuffer'是一个简单的FIFO队列实现。它只提供这两个方法的基本功能。

   1. ``push``: 在缓冲区内存入一些收集到的数据。如果超过了缓冲区的最大容量，队列头的数据将被从缓冲区中移除。
   2. ``sample``: 以随机方式对长度为`batch_size`的列表进行均匀采样。

在 "NaiveReplayBuffer "的基础上，"AdvancedReplayBuffer "和 "EpisodeReplayBuffer "分别实现了更多的功能和特性。

``AdvancedReplayBuffer`` 实现了以下功能(如图所示)：

   - **优先取样**：参考论文`<https://arxiv.org/abs/1511.05952>`_。
   - **监控数据质量（使用次数和陈旧程度）**：如果一个数据被使用的次数太多，或者过于陈旧，无法优化策略，它将被从缓冲区中移除。

   .. note::
      **use_count**: 计算一个数据被取样的次数。

      **staleness**: 采集时和采样时的模型迭代间隙
   
   - **吞吐量的监测和控制**。在一个固定的时间段内，计算有多少数据被推入、取出或移出缓冲区。在一定范围内控制 "推入"/"取出 "的比例，以防数据流速度不匹配。
   - **记录器**。采样的数据属性和吞吐量显示在文本记录器和tensorboard记录器中。

.. image::
   images/advanced_buffer.png
   :align: center
   :scale: 65%
   

.. tip::
   默认情况下，DI-engine中的大多数策略都采用 "AdvancedReplayBuffer"，因为我们认为监视器和记录器在调试和策略调整中相当重要。然而，如果你确定你不需要上述所有的功能，你可以随时切换到更简单、更快速的`NaiveReplayBuffer'。

`EpisodeReplayBuffer`是为一些特殊情况而设计的，他们需要一整集而不是分离的样本。比如说。在国际象棋、围棋或纸牌游戏中，玩家只有在游戏结束后才能得到奖励；在一些算法中，如`事后诸葛亮式的经验回放 <https://arxiv.org/abs/1707.01495>`_，必须对整个情节进行采样并进行操作。因此，在``EpisodeReplayBuffer``中，每个元素不再是一个训练样本sample，而是一个剧集episode。

在DI-engine中，我们定义了**完整的数据**和**元数据**。**完整数据**通常是一个dict，其键值为`['obs', 'action', 'next_obs', 'reward', 'info']`和一些可选的键值，如`['priority', 'use_count', ' collect_iter', ...`。然而，在一些复杂的环境中（通常我们以并行模式运行），``['obs', 'action', 'next_obs', 'reward', 'info']``可能太大，无法存储在内存中。因此，我们将它们存储在文件系统中，只在内存中存储**meta数据，包括`''file_path''和可选键。因此，在并行模式下，当从缓冲区中删除数据时，我们不仅要删除内存中的元数据，而且还要删除文件系统中的元数据。

如果你想了解有关三种缓冲区的更多细节，或并行模式下的移除机制，请参考 "缓冲区概述<.../feature/replay_buffer_overview_en.html>"_

Worker-评估器
~~~~~~~~~~~~~~~~

评估器是DI-engine的另一个关键执行组件，用于确定训练模型是否收敛。与收集器类似，评估器由三个关键组件组成--环境管理器、策略（eval_mode）、评估器控制器。

环境管理器允许我们一个一个地运行多个环境（"base_env_manager"）或并行地运行（"subprocess_env_manager"）。例如，如果我们使用子进程环境管理器，我们将在不同的子进程中运行不同的环境，这将大大增加收集数据的效率。

Policy(eval_mode)是我们需要评估的RL模型。

评估器控制器是一个决定我们是否应该停止评价的组件。例如，在 "Serial Evaluator "中，"n_evaluator_episode "是一个参数，用来确定我们要收集和评价多少个episode。一旦我们收集了这些数量的episode，评估器将停止收集并开始计算平均奖励。如果平均数大于`stop_value'，`stop_flag'将为True，我们将知道我们的模型已经收敛了。

串行评估器
^^^^^^^^^^^^^^^^^^^

串行评估器用于串行模式中。关键概念是 "n_evaluator_episode "和 "stop_value"。

下面是一个如何使用串行评估器的例子。

.. code:: python

    import gym
    from easydict import EasyDict
    from ding.policy import DQNPolicy
    from ding.env import BaseEnvManager
    from ding.worker import BaseSerialEvaluator

    # prepare components
    cfg: EasyDict  # config after `compile_config`
    normal_env = BaseEnvManager(...)  # normal vectorized env
    
    dqn_policy = DQNPolicy(cfg.policy)
    evaluator = BaseSerialEvaluator(cfg.policy.eval.evaluator, normal_env, dqn_policy.eval_mode)

    # evalulate 10 env episode
    stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep, n_episode=10)
    assert isinstance(reward, list) and len(reward) == 10

    # judge whether the return value reaches the convergence standard
    if stop:
       break

.. note::
   不同的环境可能有不同的 "stop_value "和 "n_evaluator_episode"。例如，在 cartpole我们有`stop_value=195`和`n_evaluator_episode=100`。用户应该在 "env "中注明这两个参数配置（即`env.stop_value', `env.n_evaluator_episode'），然后它们将被传递给评估器。


结合评估条件（即 "should_eval "方法），我们可以将评估器添加到串行模式中，如下所示。

.. code:: python

    for _ in range(max_iterations):
        
        # Evaluate policy performance
        if evaluator.should_eval(learner.train_iter):
            #load model
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            # if stop flag, exit the process
            if stop:
                break
            # if not stop flag, continue to collect data and train the model
            new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)


.. tip::
   **如何判断模型是否已经收敛**

   我们根据平均奖励来判断模型是否收敛。在DI-engine中，有三种类型的平均奖励：获胜概率、总累积奖励和平均单位步骤奖励。

   获胜概率。在像 "SMAC "这样的游戏中，我们关注的是最终的结果，而不太关心游戏的过程。对于这种环境，我们使用获胜概率（对于`SMAC'1.0的3s5z）作为收敛条件。

   总累积奖励。在 "cartpole "和 "Lunarlander "等游戏中，我们需要使总分尽可能大。所以我们使用总累积奖励作为收敛条件。

   平均单位步骤奖励。在一些游戏中，我们需要使总奖励尽可能大，同时减少不必要的探索步骤的数量。对于这种环境，我们使用平均单位步数奖励作为收敛条件。

   此外，一个可靠的RL实验应该用不同的随机种子重复3~5次，．同时，采用一些统计数据如中位数值和平均值/std值可能更有说服力。


.. tip::

   **如何解决评价器中的不同环境可能启动不同长度episode的问题？**
   
   在某些情况下，这确实是一个大问题。例如，假设我们想在评估器中收集12个情节，但只有5个环境，如果我们不做任何事情，很可能我们会得到更多的短episode而不是长episode。因此，我们的平均奖励会有偏差，可能不准确。这是显而易见的，因为短episode需要的时间更少。

   为了解决这个问题，我们使用 "VectorEvalMonitor"组件来平衡每个环境要收集多少个情节。让我们回到上面的例子，我们将为前两个环境中的任何一个收集三个情节，但对其余环境中的每一个只收集两个。

   此外，我们使用 "get_episode_reward "来获得每个环境中k个情节的奖励之和，以及 "get_current_episode "来获得每个环境中的情节数k。


Worker-Learner
~~~~~~~~~~~~~~~~~~
学习者是所有工作者中最重要的组件之一，他负责通过训练数据优化策略。与另一个重要的组件 "收集器 "不同，学习者不分为串行和并行模式，也就是说，只有一个学习者类，串行和并行入口可以调用不同的方法进行训练。

**串行模式**会调用学习者的`train'方法进行训练。`train'方法接收一批数据，并调用learning_mode策略的`_forward_learn'来训练一个迭代周期。

**并行模式**将调用学习者的`start'方法进行完整的训练过程。`start'方法有一个循环，包括从源头（通常是文件系统）获取数据，并调用`train'进行一次迭代训练。`start'将训练特定的迭代次数，这是由配置参数设置的。

除了`train'和`start'，学习者还提供了一个有用的接口，叫做`save_checkpoint'，它可以在训练中保存当前状态作为检查点。

在学习者中，有一个特殊的概念叫做 "Hook"（钩子）。钩子负责在特定的时间做一些固定的工作，包括 "before_run"（在``start``开始时）、"after_run"（在``start``结束时）、"before_iter"（在``train``开始时）、"after_iter"（在``train``结束时）。

钩子有很多不同的类型。DI-engine现在有钩子来保存检查点（`save_checkpoint`也使用这个钩子），加载检查点，打印日志（text & tb），减少多个学习者的日志。用户也可以轻松实现自己的钩子。如果你想了解更多关于钩子机制的信息，你可以参考`Wrapper & Hook Overview <.../feature/wrapper_hook_overview_en.html>`_。



入口
~~~~~~~~~~~~~~~~~
DI-engine提供了3个不同用途的入口，用户可以选择他们喜欢的任何一个。


串行管道Serial Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^
    
    有3种不同进入方式，用户可以在实践中选择他们喜欢的。不同的条目是为各种需求设计的。

    1. CLI
        
        **简单地运行一个训练程序，验证正确性，获得RL模型或专家数据。**

        .. code:: bash

            # usage 1(without config)
            ding -m serial -e cartpole -p dqn --train-iter 100000 -s 0
            # usage 2(with config)
            ding -m serial -c cartpole_dqn_config.py -s 0

        你可以输入``ding -h``中获取更多信息。
        
    2. 定制主函数　（Customized Main Function）

        **定制你的RL训练入口，设计算法或在你的环境中应用它。

        参考一些主函数python文件的例子，在``dizoo/envname/entry/envname_policyname_main.py``，例如。

            - dizoo/classic_control/cartpole/entry/cartpole_dqn_main.py
            - dizoo/classic_control/cartpole/entry/cartpole_ppo_main.py
            - dizoo/classic_control/pendulum/entry/pendulum_td3_main.py

        .. code:: bash

            python3 -u cartpole_dqn_main.py  # users can also add arguments list in your own entry file

    3. 统一入口函数　（Unified Entry Function）

        **配置配置文件，只需调整超参数，并在现有的算法中进行比较试验。

        .. code:: python

            from ding.entry import serial_pipeline
            from dizoo.classic_control.cartpole.config.cartpole_dqn_config import main_config, create_config
            serial_pipeline([main_config, create_config], seed=0)

        你可以参考``ding/entry``目录，阅读相关的条目功能和测试。

并行管道Parallel Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    1. CLI

    .. code:: bash
        
        # config path: dizoo/classic_control/cartpole/config/parallel/cartpole_dqn_config.py
        ding -m parallel -c cartpole_dqn_config.py -s 0

    2. 统一入口函数　（Unified Entry Function）

    .. code:: python

        from ding.entry import parallel_pipeline
        from dizoo.classic_control.cartpole.config.parallel.cartpole_dqn_config import main_config, create_config, system_config
        parallel_pipeline([main_config, create_config, system_config], seed=0)

Dist Pipeline
^^^^^^^^^^^^^^^

    1. CLI for local

    .. code:: bash

        # config path: dizoo/classic_control/cartpole/config/parallel/cartpole_dqn_config.py
        export PYTHONUNBUFFERED=1
        ding -m dist --module config -p local -c cartpole_dqn_config.py -s 0
        ding -m dist --module learner --module-name learner0 -c cartpole_dqn_config.py.pkl -s 0 &
        ding -m dist --module collector --module-name collector0 -c cartpole_dqn_config.py.pkl -s 0 &
        ding -m dist --module collector --module-name collector1 -c cartpole_dqn_config.py.pkl -s 0 &
        ding -m dist --module coordinator -p local -c cartpole_dqn_config.py.pkl -s 0

    2. CLI for server(such as SLURM)

    .. code:: bash

        # config path: dizoo/classic_control/cartpole/config/parallel/cartpole_dqn_config.py
        export PYTHONUNBUFFERED=1
        learner_host=10-10-10-10
        collector_host=10-10-10-[11-12]
        partition=test

        ding -m dist --module config -p slurm -c cartpole_dqn_config.py -s 0 -lh $learner_host -clh $collector_host
        srun -p $partition -w $learner_host --gres=gpu:1 ding -m dist --module learner --module-name learner0 -c cartpole_dqn_config.py.pkl -s 0 &
        srun -p $partition -w $collector_host ding -m dist --module collector --module-name collector0 -c cartpole_dqn_config.py.pkl -s 0 &
        srun -p $partition -w $collector_host ding -m dist --module collector --module-name collector1 -c cartpole_dqn_config.py.pkl -s 0 &
        ding -m dist --module coordinator -p slurm -c cartpole_dqn_config.py.pkl -s 0

    3. CLI for k8s

        TBD

.. tip::
  如果你想了解更多关于算法实现、框架设计和效率优化的细节，我们还提供了`特征<../feature/index.html>`_的文档。

计算模式
----------------------

Serial Pipeline
~~~~~~~~~~~~~~~~~

Off-Policy DRL: DQN, IMPALA, SAC
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image::
   images/serial_pipeline.svg
   :align: center

用户可以通过结合和利用DI-engine中不同的描述和执行模块，轻松实现各种DRL算法，这里有一些演示设计。

On-Policy DRL: PPO
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Changes**: 删除缓冲区

.. image::
   images/serial_pipeline_on_policy.svg
   :align: center

**DRL + RewardModel: GAIL, HER, RND**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Changes**: 增加奖励模型和相关的数据转换

.. image::
   images/serial_pipeline_reward_model.svg
   :align: center

**DRL + Demostration Data/Policy: R2D3, SQIL, RBC**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Changes**: 添加专家数据缓冲区（演示缓冲区）或专家策略所描述的收集器

.. image::
   images/serial_pipeline_r2d3.svg
   :align: center

Parallel/Dist Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~

**Changes**: 协调员和objstore，策略流，数据流（meta and step）和任务流

.. image::
   images/parallel_pipeline.svg
   :align: center
