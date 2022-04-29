Unit Test Guide
=========================

Significance of Unit Testing
----------------------------------------

In the field of software engineering, unit testing is a testing method. Through this method, each unit of the source code set of one or more computer program modules and related control data, use programs and operation programs are tested to determine whether they can operate correctly (from `Wikipedia - Unit testing <https://en.wikipedia.org/wiki/Unit_testing>`_).

In actual development, the significance of unit testing is as follows:

*When the code is updated, you can run unit tests to ensure that regression errors do not occur.
*Through fine-grained unit test design, you can quickly and accurately locate the source of errors during unit test.
*Combining unit testing with code coverage ensures that all code and branches have been tested.
*After finding bugs, you can add test cases that can reproduce bugs to unit tests to continuously improve the perfection of code functions.
*Another important point -- **for a module, reading unit test code is also a very efficient way to understand its function and usage**.



Types of Unit Test
---------------------------------

In the DI-engine project, we divide the unit test into the following parts:

* ``unittest`` -- functional unit test in a general sense to ensure the normal function of engineering code and the convergence of algorithm code on simple use cases.
* ``algotest`` -- unit test for algorithm code to ensure that the algorithm code can meet the use requirements on specific use cases.
* ``cudatest`` -- for unit testing of CUDA dependent features, ensure that such features function normally in the operating environment with CUDA.
* ``envpooltest`` -- unit test for features that rely on envpool high-performance parallel computing to ensure that such features function normally.
* ``platformtest`` -- for unit testing of cross platform code, ensure that the core functions of DI engine can still operate normally on MacOS and windows platforms.
* ``benchmark`` -- for the performance test of algorithm or architecture, speed measurement is mainly carried out for relevant contents to ensure that its performance meets the requirements.



How to Build Unit Test
---------------------------------



How do Do Unit Test
---------------------------------
