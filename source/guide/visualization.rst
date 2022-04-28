Graphics and Visualization
========================================

In DI-engine, we often need to draw images and measure and visualize some information. This section will introduce these contents in detail.


PlantUML
-----------------

PlantUML is a tool that can be used to draw UML and other images. For details, please refer to `the official website of PlantUML <https://plantuml.com/zh/>`_.

For example, we can draw class diagrams

.. image:: plantuml-class-demo.puml.svg
    :align: center

You can draw the flow chart of the algorithm

.. image:: plantuml-activity-en-demo.puml.svg
    :align: center

YAML data can also be plotted

.. image:: plantuml-yaml-demo.puml.svg
    :align: center

We can use plantumlcli tool to generate images. For details, please refer to `plantumlcli GitHub repository <https://github.com/HansBug/plantumlcli>`_.

.. note::

    In the document of DI engine, plantuml has been integrated, which can automatically generate images based on source code. For example, we can create the file ``plantuml-demo.puml`` under the current path.

    .. literalinclude:: plantuml-demo.puml
        :language: text
        :linenos:

    When compiling the document, the image ``plantuml-demo.puml.svg`` in SVG format will also be generated automatically, as shown below.

    .. image:: plantuml-demo.puml.svg
        :align: center



graphviz
-----------------

For more complex topology diagrams, we can use tool Graphviz to draw:

* `Official Documentation of Graphviz <https://graphviz.org/>`_
* `Python Wrapper Library of Graphviz  <https://github.com/xflr6/graphviz>`_
* `Graphviz Online <https://dreampuf.github.io/GraphvizOnline/>`_

For example, we can use graphviz to quickly draw a graph structure, as shown in the following code

.. literalinclude:: graphviz-demo.gv
    :language: text
    :linenos:

The drawn image is shown below

.. image:: graphviz-demo.svg
    :align: center



draw.io
-----------------




snakeviz
-----------------







