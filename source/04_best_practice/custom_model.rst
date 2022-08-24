如何自定义 model 模型
=================================================

policy 默认 model
----------------------------------

ding 下已实现的 policy 中均对 default_model 进行了定义，具体可看 \ `policy-default_model 链接 <https://vsde0sjona.feishu.cn/wiki/wikcnhgRDmxwU4G529aQz5BPDdh>`__\ 例如 SAC 的 default_model：

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

自定义 model 适用条件
----------------------------------

但很多时候 DI-engine 中实现的 \ ``policy``\ 中的  \ ``default_model``\ 不适用自己的任务，例如这里想要在 \ ``dmc2gym``\ 环境 \ ``cartpole-swingup``\  任务下应用 \ ``sac``\ 算法，且环境 observation 为  \ ``pixel``\ ，
即 \ ``obs_shape = (3, height, width)``\ （如果设置 \ ``from_pixel = True, channels_first = True``\ ，详情见  \ `dmc2gym 环境文档 <https://github.com/opendilab/DI-engine-docs/blob/main/source/13_envs/dmc2gym_zh.rst>`__\  

而此时查阅 \ `sac 源码 <https://github.com/opendilab/DI-engine/blob/main/ding/policy/sac.py>`__\ 可知 \ ``default_model``\ 为 \ `qac <https://github.com/opendilab/DI-engine/blob/main/ding/model/template/qac.py>`__\ ，
\ ``qac model``\ 中暂时只支持 \ ``obs_shape``\ 为一维的情况，此时我们即可根据需求自定义 model 并应用到 policy。

自定义 model 基本步骤
----------------------------------

1. 明确env policy task
   
  - 比如这里选定 sac+dmc2gym+cartpole-swingup，且要求from_pixel

2. 选择policy

  -  选择sac

3. 发现policy中的default_model不适用自己的任务
   
  -  比如这里 sac 的 defaul_model 为 qac ，但是我们的obs_shape=(3,100,100)，现有的qac只能处理obs_shape为一维的情况

4. 根据已有的defaul_model来决定所需实现的功能

  -  比如这里需要实现 init ， forward ， 包含两个actor和critic分别对应的 forward

5. 如何应用自定义模型

  -  如新pipeline是直接定义model，传入 policy 进行初始化即可
  
    .. code:: python
        
        ...
        model = QACPixel(**cfg.policy.model)
        policy = SACPolicy(cfg.policy, model=model) 
        ...


  -  旧pipeline是修改default_model（有其他方法吗）

6. 进行测试

