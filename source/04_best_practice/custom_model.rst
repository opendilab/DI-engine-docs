如何自定义 model 模型
=================================================

policy 默认 model
----------------------------------

ding 下已实现的 policy 中均对 default_model 进行了定义，具体可看 \ `policy-default_model 链接 <https://xxx>`__\ ，例如 SAC 的 default_model：

.. code:: python

   @POLICY_REGISTRY.register('sac')
    class SACPolicy(Policy):
    ...

        def default_model(self) -> Tuple[str, List[str]]:
            if self._cfg.multi_agent:
                return 'maqac_continuous', ['ding.model.template.maqac']
            else:
                return 'qac', ['ding.model.template.qac']
    ...

此处return的 \ ``'maqac_continuous', ['ding.model.template.maqac'] ``\ ，前者是 model registry 的名字，后者是 model 所处的文件路径。

自定义 model 适用条件
----------------------------------

但很多时候 DI-engine 中实现的 \ ``policy``\ 中的  \ ``default_model``\ 不适用自己的任务，例如这里想要在 \ ``dmc2gym``\ 环境 \ ``cartpole-swingup``\  任务下应用 \ ``sac``\ 算法，且环境 observation 为  \ ``pixel``\ ，
即 \ ``obs_shape = (3, height, width)``\ （如果设置 \ ``from_pixel = True, channels_first = True``\ ，详情见  \ `dmc2gym 环境文档 <https://github.com/opendilab/DI-engine-docs/blob/main/source/13_envs/dmc2gym_zh.rst>`__\ ） 

而此时查阅 \ `sac 源码 <https://github.com/opendilab/DI-engine/blob/main/ding/policy/sac.py>`__\ 可知 \ ``default_model``\ 为 \ `qac <https://github.com/opendilab/DI-engine/blob/main/ding/model/template/qac.py>`__\ ，
\ ``qac model``\ 中暂时只支持 \ ``obs_shape``\ 为一维的情况，此时我们即可根据需求自定义 model 并应用到 policy。

自定义 model 基本步骤
----------------------------------

1. 明确 env, task, policy
   
-  比如这里选定 \ ``dmc2gym``\ 环境 \ ``cartpole-swingup``\  任务，且设置 \ ``from_pixel = True, channels_first = True``\ （详情见  \ `dmc2gym 环境文档 <https://github.com/opendilab/DI-engine-docs/blob/main/source/13_envs/dmc2gym_zh.rst>`__\ ） 
   ，即此时观察空间为图像 \ ``obs_shape = (3, height, width)``\ ，并选择 \ ``sac``\ 算法进行学习。


2. 查阅 policy 中的 default_model 是否适用

-  此时根据\ `policy-default_model 链接 <https://xxx>`__\ 或者直接查阅源码 \ `ding/policy/sac:SACPolicy <https://github.com/opendilab/DI-engine/blob/main/ding/policy/sac.py>`__\ ，找到 SAC 的 default_model：

.. code:: python

   @POLICY_REGISTRY.register('sac')
    class SACPolicy(Policy):
    ...

        def default_model(self) -> Tuple[str, List[str]]:
            if self._cfg.multi_agent:
                return 'maqac_continuous', ['ding.model.template.maqac']
            else:
                return 'qac', ['ding.model.template.qac']
    ...

-  进一步查看  \ `ding/model/template/qac:QAC <https://github.com/opendilab/DI-engine/blob/69db77e2e54a0fba95d83c9411c6b11cd25beae9/ding/model/template/qac.py#L40>`__\ ，
   发现 DI-engine 中实现的 \ ``qac model``\ 暂时只支持 \ ``obs_shape``\ 为一维的情况，但是此时环境的观察空间为图像 \ ``obs_shape = (3, height, width)``\ ，
   因此我们需要根据需求自定义 model 并应用到 policy。

3. custom_model 实现

根据已有的 defaul_model 来决定 custom_model 所需实现的功能:

-  需要实现原default model中所有的方法
  
-  保证返回值的类型的原default model一致

具体实现可利用 \ `ding/model/common <https://github.com/opendilab/DI-engine/tree/main/ding/model/common>`__\ 下 \ ``encoder.py``\ / \ ``head.py``\ 已实现的 \ ``encoder``\ 和 \ ``head``\ 

- 已实现的 encoder ：

+-----------------------+-------------------------------------+
|encoder                |usage                                |
+=======================+=====================================+
|ConvEncoder            |处理图像obs输入                      |
+-----------------------+-------------------------------------+
|FCEncoder              |处理一维obs输入                      |                
+-----------------------+-------------------------------------+
|StructEncoder          |                                     |
+-----------------------+-------------------------------------+

- 已实现的 head ：

+-----------------------+-------------------------------------+
|head                   |usage                                |
+=======================+=====================================+
|DiscreteHead           |输出离散动作指                       |
+-----------------------+-------------------------------------+
|DistributionHead       |输出 q-value 分布                    |
+-----------------------+-------------------------------------+
|RainbowHead            |                                     |
+-----------------------+-------------------------------------+
|QRDQNHead              |                                     |
+-----------------------+-------------------------------------+
|QuantileHead           |                                     |
+-----------------------+-------------------------------------+
|DuelingHead            |                                     |
+-----------------------+-------------------------------------+
|RegressionHead         |                                     |
+-----------------------+-------------------------------------+
|ReparameterizationHead |                                     |
+-----------------------+-------------------------------------+
|MultiHead              |                                     |
+-----------------------+-------------------------------------+


例如这里需要自定义针对 sac+dmc2gym+cartpole-swingup 任务的 model ，我们把新的 custom_model 实现为 \ ``QACPixel``\  类

-  这里对照 \ ``QAC``\ 已经实现的方法， \ ``QACPixel``\ 需要实现 \ ``init``\  ， \ ``forward``\ ，以及 \ ``compute_actor``\ 和  \ ``compute_critic``\ 。

.. code:: python

  @MODEL_REGISTRY.register('qac')
    class QAC(nn.Module):
    ...
      def __init__(self, ...) -> None:
        ...
      def forward(self, ...) -> Dict[str, torch.Tensor]:
        ...
      def compute_actor(self, obs: torch.Tensor) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        ...
      def compute_critic(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ...

-  针对图像输入， \ ``QACPixel``\ 主要需要修改的是 \ ``init``\ 中对 \ ``self.actor``\ 和 \ ``self.critic``\ 的定义。
   可以看到 \ ``QAC``\ 中 \ ``self.actor``\ 和 \ ``self.critic``\ 的 encoder 都只是一层 nn.Linear

.. code:: python

  @MODEL_REGISTRY.register('qac')
  class QAC(nn.Module):
  ...
    def __init__(self, ...) -> None:
      ...
      self.actor = nn.Sequential(
              nn.Linear(obs_shape, actor_head_hidden_size), activation,
              ReparameterizationHead(
                  ...
              )
          )
      ...
      self.critic = nn.Sequential(
              nn.Linear(critic_input_size, critic_head_hidden_size), activation,
              RegressionHead(
                  ...
              )
          )

-  我们通过定义encoder_cls指定encoder的类型，加入 \ ``ConvEncoder``\ ，并且因为需要对 obs 进行encode 后和 action 进行拼接，
   将 \ ``self.critic``\ 分为  \ ``self.critic_encoder``\ 和 \ ``self.critic_head``\ 两部分

.. code:: python

  @MODEL_REGISTRY.register('qac_pixel')
  class QACPixel(nn.Module):
  def __init__(self, ...) -> None:
      ...
      encoder_cls = ConvEncoder
      ...
      self.actor = nn.Sequential(
            encoder_cls(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type),
            ReparameterizationHead(
                ...
            )
        )
      ...
      self.critic_encoder = global_encoder_cls(obs_shape, encoder_hidden_size_list, activation=activation,
                                                     norm_type=norm_type)
      self.critic_head = RegressionHead(
          ...
      )
      self.critic = nn.ModuleList([self.critic_encoder, self.critic_head])

-  再对 \ ``compute_actor``\ 和  \ ``compute_critic``\ 分别进行修改即可。

1. 如何应用自定义模型

  -  如新pipeline是直接定义model，传入 policy 进行初始化即可
  
    .. code:: python
        
        ...
        model = QACPixel(**cfg.policy.model)
        policy = SACPolicy(cfg.policy, model=model) 
        ...


  -  旧pipeline
  
    -  修改相应policy中的default_model
  
    -  通过在https://github.com/opendilab/DI-engine/blob/main/ding/entry/serial_entry.py#L22 这里传入 model, 
       在这里https://github.com/opendilab/DI-engine/blob/main/ding/entry/serial_entry.py#L59 被调用

5. 进行测试

-  todo: 详细写一下如何写test，如何启动测试，如何评价测试结果


文档问题
----------------------------------
1. encoder 和 head 的介绍有点不知道怎么写

2. “如何通过encoder_cls指定encoder的类型”？