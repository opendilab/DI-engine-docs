Code Style Guide
==========================

In `DI-engine <https://github.com/opendilab/DI-engine>`_, we use `yapf <https://github.com/google/yapf>`_ and `flake8 <https://github.com/PyCQA/flake8>`_ for code style checking and automatic repair.

yapf
-------------------

For `yapf <https://github.com/google/yapf>`_, we can use existing `Makefile <https://github.com/opendilab/DI-engine/blob/main/Makefile>`_ for one-click fix

.. code-block:: shell

   make format


Considering the large scale of the whole project and the large number of files, you can use the following commands to check the code design of the source code files in a specific path

.. code-block:: shell

   make format RANGE_DIR=./ding/xxx


In this project, we use the `yapf <https://github.com/opendilab/DI-engine/blob/main/.style.yapf>`_ code specification configuration based on PEP8. For details about the configuration, you can refer to `the description on the Github homepage <https://github.com/google/yapf#knobs>`_. `PEP8 <https://peps.python.org/pep-0008/>`_ is the code style configuration officially recommended by Python. Paying attention to code style can improve the readability of the code and minimize unintended behavior.

In addition, yapf can also integrate with pycharm through the plug-in yapf pycharm:

* `yapf-pycharm <https://plugins.jetbrains.com/plugin/9705-yapf-pycharm>`_


flake8
-------------------

For `flake8 <https://github.com/PyCQA/flake8>`_, we can use the existing `Makefile <https://github.com/opendilab/DI-engine/blob/main/Makefile>`_ to check the code design

.. code-block:: shell

   make flake_check


Considering the large scale of the whole project and the large number of files, you can use the following commands to check the code design of the source code files in a specific path

.. code-block:: shell

   make flake_check RANGE_DOR=./ding/xxx


In this project, we use `flake8 code design specification configuration <https://github.com/opendilab/DI-engine/blob/main/.flake8>`_ based on pep8. For details of configuration, please refer to `the description of flake8 official documents <https://flake8.pycqa.org/en/latest/user/configuration.html>`_. `PEP8 <https://peps.python.org/pep-0008/>`_ is the code style configuration officially recommended by python. Paying attention to the code style can improve the readability of the code and minimize the behavior that does not meet the expectations.



