安装说明
===============================

.. toctree::
   :maxdepth: 2

前置条件
--------------

系统版本: Linux, macOS, Windows
   
Python 版本: 3.6-3.8 

.. note::

    如果您的设备具有 Nvidia GPU ，参考 `Nvidia CUDA Toolkit 安装 <https://developer.nvidia.com/cuda-downloads/>`_。 

    在安装好 CUDA 之后，当您在安装 DI-engine 的依赖项时，会自动获取和安装带有 Nvidia CUDA 加速的 PyTorch。
    
    如您需要手动安装适合本机 Python 版本和 GPU 设备的 PyTorch 版本，您可以参考 `PyTorch 安装 <https://pytorch.org/get-started/locally/>`_ 。

    此外，如果您的操作系统是 Windows ，请您检查是否拥有 SWIG 的 Windows 应用程序，您可以参考 `SWIG 安装 <https://www.swig.org/download.html>`_ ，并将SWIG的可执行文件添加至 Windows 系统环境变量 PATH 路径中。

发布版本
--------------

您可以使用以下命令安装 DI-engine 的稳定发行版：

.. code-block:: bash

    # Install the latest pip
    pip install
    # Current stable release of DI-engine
    pip install DI-engine

.. tip::

    如果您在下载软件包时遇到超时错误，您可以尝试采用更换其他常用镜像源，例如使用阿里源：

    .. code-block:: bash

        pip install requests -i https://mirrors.aliyun.com/pypi/simple/ DI-engine    

如果您需要借助 Anaconda 或者 Miniconda , 建议您采用如下命令:

.. code-block:: bash

    conda install -c opendilab di-engine


开发版本
--------------

如果您需要从 Github 源码安装最新的 DI-engine 开发版本，可以使用如下方法：

.. code-block:: bash

    git clone https://github.com/opendilab/DI-engine.git
    pip install ./DI-engine/

特殊版本
--------------

如果您希望启用 DI-engine 额外功能并安装相关依赖，可以使用如下方法：

.. code-block:: bash

    # install atari and box-2d related packages
    pip install DI-engine[common_env]
    # install unittest(pytest) related packages
    pip install DI-engine[test]
    # enable numba acceleration
    pip install DI-engine[fast]
    #install multi extra packages
    pip install DI-engine[common_env,test,fast]

使用 Docker 运行
--------------

DI-engine 的镜像可以在 `DockerHub <https://hub.docker.com/r/opendilab/ding>`_ 获得。拉取镜像的方法如下：

.. code-block:: bash

    # Download Stable release DI-engine Docker image
    docker pull opendilab/ding:nightly 
    # Run Docker image
    docker run -it opendilab/ding:nightly /bin/bash
