Installation Guide
============================

.. toctree::
   :maxdepth: 2

Prerequisites
--------------

Operation system version: Linux, macOS, Windows

Python version: 3.6-3.8 

.. note::

    If there is a GPU in your setting, you can refer to `Nvidia CUDA Toolkit Installation <https://developer.nvidia.com/cuda-downloads/>`_
    
    After CUDA being installed, you will get a correct Nvidia CUDA version of Pytorch automaticly when installing DI-engine.
    
    If you want to install Pytorch manually, you can refer to `PyTorch Installation <https://pytorch.org/get-started/locally/>`_.

    If your OS is Windows, please do confirm that SWIG is installed and available through the OS environment variable PATH, you can refer to `SWIG installation <https://www.swig.org/download.html>`_.

Stable Release Version
--------------

You can simply install stable release DI-engine with the following command:

.. code-block:: bash

    # Install the latest pip
    pip install
    # Current stable release of DI-engine
    pip install DI-engine

.. tip::

    If you encounter timeout in downloading packages, you can try to request from other site.

    .. code-block:: bash

        pip install requests -i https://mirrors.aliyun.com/pypi/simple/ DI-engine    

And if you prefer to use Anaconda or Miniconda, the following command is suggested:

.. code-block:: bash

    conda install -c opendilab di-engine

Development Version
--------------

If you need to install latest DI-engine in development from the Github source codes:

.. code-block:: bash

    git clone https://github.com/opendilab/DI-engine.git
    pip install ./DI-engine/

Special Version
--------------

If you want to enable special version of DI-engine and install the extra packages that are required, you can use the following command:

.. code-block:: bash

    # install atari and box-2d related packages
    pip install DI-engine[common_env]
    # install unittest(pytest) related packages
    pip install DI-engine[test]
    # enable numba acceleration
    pip install DI-engine[fast]
    #install multi extra packages
    pip install DI-engine[common_env,test,fast]


Run in Docker
--------------

DI-engine docker images are available in `DockerHub <https://hub.docker.com/r/opendilab/ding>`_. You can use the following commands to pull the image:

.. code-block:: bash

    # Download Stable release DI-engine Docker image
    docker pull opendilab/ding:nightly 
    # Run Docker image
    docker run -it opendilab/ding:nightly /bin/bash
