### R2D3

### 概述

R2D3 (Recurrent Replay Distributed DQN from Demonstrations) 首次在论文
[Making Efficient Use of Demonstrations to Solve Hard Exploration Problems][r2d3-paper] 中提出,
它可以有效地利用专家演示来解决初始条件高度可变、部分可观察环境中的困难探索问题。 
此外他们还介绍了一组结合这三个属性的八个任务，并表明 R2D3 可以解决像这样的任务，注意的是在这些任务上，其他一些最先进的方法，无论有还是没有专家演示，在数百亿次的探索步骤之后甚至可能
仍然无法看到一条成功轨迹。r2d3本质上是有效利用了r2d2算法的分布式框架和循环神经结构以及DQfD的为从专家轨迹中学习而特别设计的损失函数。


### 核心要点
1. R2D3的基线强化学习算法是 [R2D2][R2D2] ,可以参考我们的实现
   [r2d2][r2d2] ，它本质上是一个基于分布式框架，采用了双Q网络, dueling结构，n-step td的DQN算法。

2. R2D3利用了DQfD的损失函数，包括一步和n步的TD损失，网络参数的L2正则化损失，监督大边际分类损失(supervised large margin classification loss),区别在于R2D3
的所有值都是在序列样本上从循环神经网络中计算得到的。

3.由于基线算法R2D2是在样本序列上进行运算的，所以我们的专家轨迹也应该以样本序列的方式给出，在实现中，往往我们是用另一个基线强化学习算法(如ppo)收敛后得到的专家模型来
产生对应的专家演示，为此我们专门写了对应的策略函数来从像ppo这样的专家模型中产生专家演示，参见[ppo_offpolicy_collect_traj.py][ppo_offpolicy_collect_traj.py]。

4.用于训练Q网络的mini-batch里以pho的概率是专家演示序列，1-pho的概率是智能体与环境交互的经验序列，

5. r2d3e的提出是为了解决初始条件高度可变、部分可观察环境中的困难探索问题，其他探索相关的论文，读者可以参考 
[NGU][NGU]，它是融合了 [ICM][ICM] 和 [RND][RND] 等多种探索方法的一个综合体。

### 关键方程或关键框图

 R2D3算法的整体分布式训练流程如下：
<img src = '/Users/puyuan/code/DI-engine-docs/source/hands_on/images/r2d3_overview.png' style="zoom:20" />
其中learner用于训练的mini_batch包含了2部分：1，专家演示轨迹，2，智能体在训练过程中与环境交互产生的经验轨迹。 演示和智能体经验验之间的比率是一个关键的超参数，必须仔细调整以实现良好的性能。

R2D3算法的整Q网络结构图如下：
<img src = '/Users/puyuan/code/DI-engine-docs/source/hands_on/images/r2d3_q_net.png' style="zoom:20" />
(a)R2D3智能体使用的recurrent head。 (b) DQfD智能体使用的feedforward head
（C）表示输入为大小为 96x72 的图像帧，接着通过一个ResNet，然后将前一时刻的动作，前一时刻的奖励和当前时刻的其他本体感受特征f_t
（包括加速度、avatar是否握住物体以及手与avatar的相对距离等辅助信息）连接(concat)为一个新的输入向量，传入a)和b)中的head，用于计算Q值。

```latex
$f_t$
```
```latex
$\text{在上面的式子中 } \psi \text{ 是convex cost function regularizer。} \text{加入此项的目的在于， 如果我们从有限的数据集中学习一个cost function，如果这个cost function的capacity非常大，非常会出现overfitting的情况。}\\ \text{通过使用不同的IRL regulariser } \psi, \text{ 此框架可以生成不同的算法。 例如，当 } \psi \text{ 的定义如下时，我们得到了文章的主角， GAIL算法} $
```
下面描述r2d3的损失函数设置，和DQfD一样，不过这里所有的Q值都是通过上面所述的循环神经网络结构计算得到。

除了通常的1-step turn,r2d3还增加n-step turn，有助于将专家轨迹的值传播到所有的早期状态，从而获得更好的预训练效果。 n步return为：
<img src = '/Users/puyuan/code/DI-engine-docs/source/hands_on/images/r2d3_nstep_return.png' style="zoom:20" />

监督损失对于预训练的效果至关重要。 由于演示数据可能只是覆盖状态空间的一小部分并且在某一个状态并没有采取所有可能的动作，
因此许多状态动作对从未被专家采取。 如果我们仅使用 Q-learning 朝着下一个状态的最大Q值来更新预训练网络，网络将倾向于朝着那些不准确的变量中的最高值方向更新，
并且网络将在整个过程中通过Q函数传播这些值。
监督大边际分类损失(supervised large margin classification loss)公式：
<img src = '/Users/puyuan/code/DI-engine-docs/source/hands_on/images/r2d3_slmcl.png' style="zoom:20" />
其中a_E表示专家执行的动作。

我们在DI-engine中的具体实现如下所示：
```latex
$l = margin_function * torch.ones_like(q)$
$l.scatter_(1, torch.LongTensor(action.unsqueeze(1)), torch.zeros_like(q))$
$JE = is_expert * (torch.max(q + l.to(device), dim=1)[0] - q_s_a)$
```
r2d3还添加了应用于网络权重和偏差的 L2 正则化损失，以帮助防止它在相对数量较小的演示数据集上过度拟合。
最终用于更新网络的整体损失是所有四种损失的组合：
<img src = '/Users/puyuan/code/DI-engine-docs/source/hands_on/images/r2d3_loss.png' style="zoom:20" />

### 伪代码

下面是包含R2D3智能体learner和actor的伪代码。 智能体由单个学习器进程(learner process)组成
从专家演示缓冲区和智能体经验缓冲区中采样以更新其策略参数。
智能体还包含A个并行的行动者进程(actor process)，这些进程与环境的副本交互以获得数据，然后将数据放入智能体经验缓冲区。 
智能体会定期更新其网络参数以匹配学习器上正在更新的参数。

<img src = '/Users/puyuan/code/DI-engine-docs/source/hands_on/images/r2d3_pseudo_code_learner.png' style="zoom:20" />
<img src = '/Users/puyuan/code/DI-engine-docs/source/hands_on/images/r2d3_pseudo_code_actor.png' style="zoom:20" />


重要的实现细节
-----------


1. demo比率pho的具体实现方式，我们是按下面的方式，获取在这个batch中专家演示所占的个数。


    # The hyperparameter pho, the demo ratio, control the propotion of data coming
    # from expert demonstrations versus from the agent's own experience.
    expert_batch_size = int(
        np.float32(np.random.rand(learner.policy.get_attribute('batch_size')) < cfg.policy.collect.pho
                   ).sum()
    )
    agent_batch_size = (learner.policy.get_attribute('batch_size')) - expert_batch_size
    train_data_agent = replay_buffer.sample(agent_batch_size, learner.train_iter)
    train_data_expert = expert_buffer.sample(expert_batch_size, learner.train_iter)


2.由于基线算法r2d2是有优先级的采样，对于一个sequence样本，我们使用TD error(1步和n步的和)绝对值，在这个序列经历的所有时刻中的平均值和最大值的加权和
作为整个序列样本的优先级。 在r2d3中我们有2个replay_buffer, 专家演示的``expert_buffer``，和智能体经验的``replay_buffer``
，为简单明了，我们是分开进行优先级采样和相关参数的更新.

    # using the mixture of max and mean absolute n-step TD-errors as the priority of the sequence
    td_error_per_sample = 0.9 * torch.max(
        torch.stack(td_error), dim=0
    )[0] + (1 - 0.9) * (torch.sum(torch.stack(td_error), dim=0) / (len(td_error) + 1e-8))
    # td_error shape list(<self._unroll_len_add_burnin_step-self._burnin_step-self._nstep>, B), for example, (75,64)
    # torch.sum(torch.stack(td_error), dim=0) can also be replaced with sum(td_error)
    ...
    if learner.policy.get_attribute('priority'):
        # When collector, set replay_buffer_idx and replay_unique_id for each data item, priority = 1.\
        # When learner, assign priority for each data item according their loss
        learner.priority_info_agent = deepcopy(learner.priority_info)
        learner.priority_info_expert = deepcopy(learner.priority_info)
        learner.priority_info_agent['priority'] = learner.priority_info['priority'][0:agent_batch_size]
        learner.priority_info_agent['replay_buffer_idx'] = learner.priority_info['replay_buffer_idx'][
            0:agent_batch_size]
        learner.priority_info_agent['replay_unique_id'] = learner.priority_info['replay_unique_id'][
            0:agent_batch_size]
    
        learner.priority_info_expert['priority'] = learner.priority_info['priority'][agent_batch_size:]
        learner.priority_info_expert['replay_buffer_idx'] = learner.priority_info['replay_buffer_idx'][
            agent_batch_size:]
        learner.priority_info_expert['replay_unique_id'] = learner.priority_info['replay_unique_id'][
            agent_batch_size:]
    
        # Expert data and demo data update their priority separately.
        replay_buffer.update(learner.priority_info_agent)
        expert_buffer.update(learner.priority_info_expert)

3.对于专家演示样本和智能体经验样本，我们分别对原数据增加一个键``is_expert``加以区分,如果是专家演示样本，此键值为1，
如果是智能体经验样本，此键值为0，
        
    # 如果是专家演示样本，此键值为1，
    for i in range(len(expert_data)):
        # for rnn/sequence-based alg.
        expert_data[i]['is_expert'] = [1] * expert_cfg.policy.collect.unroll_len  
    ...
    # 如果是智能体经验样本，此键值为0
    for i in range(len(new_data)):
        new_data[i]['is_expert'] = [0] * expert_cfg.policy.collect.unroll_len

4. 预训练，在智能体与环境交互之前，我们可以先利用专家演示样本训练Q网络，期望能得到一个好的初始参数，加速后续的训练进程。


实现
---------------

r2d3的策略``R2D3Policy`` 的接口定义如下：

.. autoclass:: ding.policy.r2d3.R2D3Policy
   :members: __init__, _forward_learn
   :noindex:

dqfd的损失函数``nstep_td_error_with_rescale``的接口定义如下：

.. autoclass:: ding.ding.rl_utils.td.dqfd_nstep_td_error_with_rescale
[comment]: <> "   :members: __init__, estimate"
   :noindex:

注意我们目前的r2d3策略实现中 网络的输入只是时刻t的状态观测，不包含时刻t-1的动作和奖励,也不包括额外的信息向量f_t.

注意：`` ... `` 表示省略的代码片段。


基准算法性能
---------

我们在PongNoFrameskip-v4做了不同的相融实验，以验证，预训练，专家演示所占比例，1步td损失，l2正则化损失等不同参数设置对算法最终性能的影响。

- PongNoFrameskip-v4（0.5M env step下，平均奖励大于0.95）

   - PongNoFrameskip-v4 + r2d3
     .. image:: images/pong_r2d3.png
     :align: center

    

- PitfallNoFrameskip-v4（10M env step下，平均奖励大于0.6）

   - MiniGrid-FourRooms-v0 +  ngu
     .. image:: images/fourrooms_ ngu.png
     :align: center
   


参考资料
---------
1.Paine T L, Gulcehre C, Shahriari B, et al. Making efficient use of demonstrations to solve hard exploration problems[J]. arXiv preprint arXiv:1909.01387, 2019.

2.Kapturowski S, Ostrovski G, Quan J, et al. Recurrent experience replay in distributed reinforcement learning[C]//International conference on learning representations. 2018.

3.Badia A P, Sprechmann P, Vitvitskyi A, et al. Never give up: Learning directed exploration strategies[J]. arXiv preprint arXiv:2002.06038, 2020.

4.Burda Y, Edwards H, Storkey A, et al. Exploration by random network distillation[J]. https://arxiv.org/abs/1810.12894v1. arXiv:1810.12894, 2018.

5.Pathak D, Agrawal P, Efros A A, et al. Curiosity-driven exploration by self-supervised prediction[C]//International conference on machine learning. PMLR, 2017: 2778-2787.





[r2d2]: https://github.com/opendilab/DI-engine/blob/main/ding/policy/r2d2.py
[R2D2]:https://openreview.net/forum?id=r1lyTjAqYX
[r2d3-paper]:https://arxiv.org/abs/1909.01387
[ppo_offpolicy_collect_traj.py]:https://github.com/opendilab/DI-engine/blob/main/ding/policy/ppo_offpolicy_collect_traj.py
[NGU]:https://arxiv.org/abs/2002.06038
[ICM]:https://arxiv.org/pdf/1705.05363.pdf
[RND]:https://arxiv.org/abs/1810.12894v1