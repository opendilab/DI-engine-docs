代码风格指南
=======================

在`DI-engine <https://github.com/opendilab/DI-engine>`_中，我们使用`yapf <https://github.com/google/yapf>`_与`flake8 <https://github.com/PyCQA/flake8>`_进行代码风格的检查与自动修复。

yapf
-------------------

对于`yapf <https://github.com/google/yapf>`_，我们可以使用现有的`Makefile <https://github.com/opendilab/DI-engine/blob/main/Makefile>`_进行一键修复

.. code-block:: shell

   make format


考虑到整个项目规模较大，文件数量较多，因此可以使用下列命令对特定路径下的源代码文件进行代码风格一键修复

.. code-block:: shell

   make format RANGE_DIR=./ding/xxx


在该项目中，我们使用基于PEP8的`yapf代码规范配置 <https://github.com/opendilab/DI-engine/blob/main/.style.yapf>`_，关于配置的详细信息，可以参考`Github主页的描述 <https://github.com/google/yapf#knobs>`_。`PEP8 <https://peps.python.org/pep-0008/>`_为Python官方推荐的代码风格配置，对代码风格的注重可以提高代码的可读性，也可以最大限度减少不符合预期的行为。

此外，yapf还可以通过插件yapf-pycharm与PyCharm进行集成：

* `yapf-pycharm <https://plugins.jetbrains.com/plugin/9705-yapf-pycharm>`_


flake8
-------------------

对于`flake8 <https://github.com/PyCQA/flake8>`_，我们可以使用现有的`Makefile <https://github.com/opendilab/DI-engine/blob/main/Makefile>`_进行代码设计上的检查

.. code-block:: shell

   make flake_check


考虑到整个项目规模较大，文件数量较多，因此可以使用下列命令对特定路径下的源代码文件进行代码设计上的检查

.. code-block:: shell

   make flake_check RANGE_DOR=./ding/xxx


在该项目中，我们使用基于PEP8的`flake8代码设计规范配置 <https://github.com/opendilab/DI-engine/blob/main/.flake8>`_，关于配置的详细信息，可以参考`flake8官方文档的描述 <https://flake8.pycqa.org/en/latest/user/configuration.html>`_。`PEP8 <https://peps.python.org/pep-0008/>`_为Python官方推荐的代码风格配置，对代码风格的注重可以提高代码的可读性，也可以最大限度减少不符合预期的行为。



