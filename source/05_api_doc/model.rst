ding.model
----------

Common
========
Please refer to ``ding/model/common`` for more details.

create_model
~~~~~~~~~~~~~
.. autofunction:: ding.model.create_model

ConvEncoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.ConvEncoder
    :members: __init__, forward

FCEncoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.FCEncoder
    :members: __init__, forward


IMPALAConvEncoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.IMPALAConvEncoder
    :members: __init__

DiscreteHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.DiscreteHead
    :members: __init__, forward


DistributionHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.DistributionHead
    :members: __init__, forward


RainbowHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.RainbowHead
    :members: __init__, forward


QRDQNHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.QRDQNHead
    :members: __init__, forward

QuantileHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.QuantileHead
    :members: __init__, quantile_net, forward

DuelingHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.DuelingHead
    :members: __init__, forward

RegressionHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.RegressionHead
    :members: __init__, forward

ReparameterizationHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.ReparameterizationHead
    :members: __init__, forward

MultiHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.MultiHead
    :members: __init__, forward

Template
========
Please refer to ``ding/model/template`` for more details.


Wrapper
=======
Please refer to ``ding/model/wrapper`` for more details.
