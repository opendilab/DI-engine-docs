worker.collector.base_serial_evaluator
==================================================

base_serial_evaluator
--------------------------

Please Reference ding/worker/collector/base_serial_evaluator.py for usage

ISerialEvaluator
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.worker.collector.base_serial_evaluator.ISerialEvaluator
    :members: default_config, __init__, reset_env, reset_policy, reset, close, should_eval, eval

VectorEvalMonitor
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.worker.collector.base_serial_evaluator.VectorEvalMonitor
    :members: __init__, is_finished, update_info, update_reward, get_episode_return, get_latest_reward, get_current_episode, get_episode_info
