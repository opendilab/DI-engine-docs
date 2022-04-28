Graphics and Visualization
========================================

In DI-engine, we often need to draw images and measure and visualize some information. This section will introduce these contents in detail.


PlantUML
-----------------

PlantUML is a tool that can be used to draw UML and other images. For details, please refer to `the official website of PlantUML <https://plantuml.com/zh/>`_.

For example, we can draw class diagrams

.. figure:: plantuml-class-demo.puml.svg
    :alt: plantuml-class-demo.puml.svg

You can draw the flow chart of the algorithm

.. figure:: plantuml-activity-en-demo.puml.svg
    :alt: plantuml-activity-en-demo.puml.svg

YAML data can also be plotted

.. figure:: plantuml-yaml-demo.puml.svg
    :alt: plantuml-yaml-demo.puml.svg

We can use plantumlcli tool to generate images. For details, please refer to `plantumlcli GitHub repository <https://github.com/HansBug/plantumlcli>`_.

.. note::

    In the document of DI engine, plantuml has been integrated, which can automatically generate images based on source code. For example, we can create the file ``plantuml-demo.puml`` under the current path.

    .. literalinclude:: plantuml-demo.puml
        :language: text
        :linenos:

    When compiling the document, the image ``plantuml-demo.puml.svg`` in SVG format will also be generated automatically, as shown below.

    .. figure:: plantuml-demo.puml.svg
        :alt: plantuml-demo.puml.svg



graphviz
-----------------




draw.io
-----------------




snakeviz
-----------------







