图像与可视化
========================================

在DI-engine中，我们经常会需要进行图像的绘制，以及一些信息的度量与可视化，本节将对这些内容进行详细介绍。


PlantUML
-----------------

PlantUML是一种可以用于绘制UML等图像的工具，详情可参考`PlantUML官方网站 <https://plantuml.com/zh/>`_。

例如，我们可以绘制类图

.. image:: plantuml-class-demo.puml.svg
    :align: center

可以绘制算法的流程图

.. image:: plantuml-activity-zh-demo.puml.svg
    :align: center

也可以绘制YAML数据

.. image:: plantuml-yaml-demo.puml.svg
    :align: center

我们可以使用plantumlcli工具进行图像的生成，具体可参考`plantumlcli的Github仓库 <https://github.com/HansBug/plantumlcli>`_。

.. note::

    而在DI-engine的文档中，已经集成了PlantUML，可以基于源代码自动生成图像。例如，我们可以在当前路径下创建文件``plantuml-demo.puml``

    .. literalinclude:: plantuml-demo.puml
        :language: text
        :linenos:

    当编译文档时，SVG格式的图像``plantuml-demo.puml.svg``也将会自动生成，如下所示。

    .. image:: plantuml-demo.puml.svg
        :align: center



graphviz
-----------------

对于更加复杂的拓扑结构图，我们可以使用graphviz工具进行绘制：

* `Graphviz官方文档 <https://graphviz.org/>`_
* `Graphviz Python封装库 <https://github.com/xflr6/graphviz>`_
* `Graphviz在线绘制 <https://dreampuf.github.io/GraphvizOnline/>`_

例如，我们可以使用Graphviz，快速绘制一个图结构，如下代码所示

.. literalinclude:: graphviz-demo.gv
    :language: text
    :linenos:

绘制的图像如下所示

.. image:: graphviz-demo.svg
    :align: center



draw.io
-----------------




snakeviz
-----------------




