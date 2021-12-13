# 分布式

当我们编写好一个 RL 训练任务时，下一个需要关心问题可能就是让它跑的更快。除了依靠算法和编译优化来让代码运行的更快，DI-engine 还设计了一套独特的横向扩展方式，让你的代码可以无缝的扩展到更多的 CPU, GPU 或者多机上面。


## Task 对象

首先假设你已经有了这样一段代码（如果没有请回到[快速开始](../quick_start/index_zh.html)）：

```python
from ding.rl_utils import get_epsilon_greedy_fn

def main():
    # Init instances
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = SampleCollector(
        cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name
    )
    evaluator = BaseSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    replay_buffer = AdvancedReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)
    # DQN training loop
    eps_cfg = cfg.policy.other.eps
    epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)
    max_iterations = int(1e8)
    for _ in range(max_iterations):
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        eps = epsilon_greedy(collector.envstep)
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs={'eps': eps})
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        for i in range(cfg.policy.learn.update_per_collect):
            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            if train_data is not None:
                learner.train(train_data, collector.envstep)
```

现在我们将介绍新引入的 `task` 对象，这是我们用于分布式扩展的基础，请将上述循环中的代码用一个方法封装起来，并放到 `task` 中：

```python
from ding.rl_utils import get_epsilon_greedy_fn
from ding.framework import Task

def training(learner, collector, evaluator, replay_buffer, epsilon_greedy):  # 1

    def _training(ctx):
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        eps = epsilon_greedy(collector.envstep)
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs={'eps': eps})
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        for i in range(cfg.policy.learn.update_per_collect):
            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            if train_data is not None:
                learner.train(train_data, collector.envstep)

    return _training

def main():
    # Init instances
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = SampleCollector(
        cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name
    )
    evaluator = BaseSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    replay_buffer = AdvancedReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)
    # DQN training loop
    eps_cfg = cfg.policy.other.eps
    epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)
    max_iterations = int(1e8)

    # Use task instead of loop
    with Task() as task:
        task.use(training(learner, collector, evaluator, replay_buffer, epsilon_greedy))
        task.run(max_step=max_iteration)
```

(1) 我们在此处使用了闭包函数来创造 `task` 需要的执行方法，这是为了让示例更加简单，你也可以使用类或其他任何能创造方法的形式，task 真正需要的只是内部这个以 ctx 为参数的方法。

在 `task` 中我们提供了一个 `use` 方法，对于熟悉某些 web 框架的开发者来说会觉得非常熟悉，例如在 gin, koa 中这是使用`中间件`的一种方式，我们的本意就是让这些拆分开的方法像真正的中间件一样，
可重复利用，甚至不仅仅在当前的任务中，你甚至可以将它封装成一个函数库，给其他开发者使用。我们希望这种方式能成为一种对开发者友好的扩展方式，能让更多的人参与到 RL 社区的贡献中。

言归正传，RL 训练中必然包含这一个无限重复的循环，我们将问题简化成每次循环是等价的，这样你只需要关注一次循环中做的事情即可。我们将一次循环的生命周期分为“采集-训练-评估”等多个阶段，你也可以加入
更多的阶段，这些阶段将会组成我们 `task` 中的最小可执行单元，即一个中间件。

接下来我们看看上面的 `_training` 函数，试着将它拆成 `evaluate`, `collect`, `train` 三个函数：

```python
from ding.rl_utils import get_epsilon_greedy_fn
from ding.framework import Task

def evaluate(learner, collector, evaluator):
    def _evaluate(ctx):
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
    return _evaluate

def collect(epsilon_greedy, learner, collector, replay_buffer):
    def _collect(ctx):
        eps = epsilon_greedy(collector.envstep)
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs={'eps': eps})
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
    return _collect

def train(learner, collector, replay_buffer):
    def _train(ctx):
        for i in range(cfg.policy.learn.update_per_collect):
            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            if train_data is not None:
                learner.train(train_data, collector.envstep)

    return _train

def main():
    # Init instances
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = SampleCollector(
        cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name
    )
    evaluator = BaseSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    replay_buffer = AdvancedReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)
    # DQN training loop
    eps_cfg = cfg.policy.other.eps
    epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)
    max_iterations = int(1e8)

    # Seperate into different middleware
    with Task() as task:
        task.use(evaluate(learner, collector, evaluator))
        task.use(collect(epsilon_greedy, learner, collector, replay_buffer))
        task.use(train(learner, collector, replay_buffer))
        task.run(max_step=max_iteration)
```

这段代码看起来能运行，但是各个中间件之间的耦合实在是有些麻烦，它让代码既难读又难改，这个时候我们一直没提到的 `ctx` 就该出场了






## 魔法时间
