Imitation learning
==================

Problem Definition and Research Motivation
----------

Imitation Learning (IL) refers to a learning method in which an agent extracts knowledge by learning some expert data, and then replicates the behavior of these expert data. Due to the inherent characteristics of IL,
It faces two major problems: it requires a lot of training data, and the quality of the training data must be good. In order to solve the above problems, in general, IL can be divided into three directions: IRL (inverse reinforcement learning), BC (behavioral cloning),
Adversarial Structured IL. The following is a brief analysis of each direction:



Research Direction
--------

BC
~~~~~~~~

BC was first proposed in [1], which proposes a supervised learning method, which directly establishes the state-action mapping relationship by fitting expert data.

The biggest advantage of BC is that it is very efficient and the algorithm is simple, but once the agent encounters a state it has never seen before, it may make the wrong behavior - this problem is called "state drift". In order to solve this problem, the DAgger [2] method adopts a method of dynamically updating the data set, and continuously adds new expert data to the data set according to the real state encountered by the training policy. In the follow-up research, IBC [3] adopted the method of implicit behavior cloning, the key of which is to train a neural network to accept observations and actions, and output a number, which is low for expert actions high for non-expert actions, thus turning behavioral cloning into an energy-based modeling problem.

The current research hotspots of BC algorithms mainly focus on two aspects: meta-learning and behavior cloning using VR devices.


IRL
~~~~~~~~

The main goal of IRL is to solve the problem of finding enough high-quality data during data collection. Specifically, IRL first learns a reward function from expert data, and then uses this reward function for subsequent RL training. With such an approach, IRL can theoretically outperform expert data.

From the specific work above, Ziebart et al. [4] first proposed the maximum entropy IRL, which utilizes the maximum entropy distribution to obtain good prospects and effective optimization. Later in 2016, Finn et al. [5] proposed a model-based approach to IRL called Guided Cost Learning (guided cost learning), this method uses a neural network to represent the cost to improve the expressive ability. Subsequently, Hester et al. proposed DQfD [6], which requires only a small amount of expert data, and significantly speeds up training through the pre-training startup process and subsequent learning process. Later methods such as T-REX [7] proposed a structure based on ranking expert data, which indirectly learned the reward function by comparing which expert data performed better.


Adversarial Structured IL
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main goal of the Adversarial Structured IL approach is to address the efficiency of IRL. It can be seen from the algorithm of IRL that even if it learns a very good reward function, since the final strategy still needs to perform reinforcement learning steps, if the strategy can be learned directly from expert data, the efficiency can be greatly improved. Based on this idea GAIL
[8] combines generative network GAN and maximum entropy IRL, which can be continuously and efficiently trained without the need to manually label expert data continuously.

On this basis, many works have improved GAIL. eg InfoGail[9] replaced GAN with WGAN and achieved better results. There are also some recent works such as GoalGAIL [10], TRGAIL [11] and DGAIL [12] which combine other methods such as post-hoc relabeling and DDPG to achieve faster convergence and better final performance.


Future Study
--------

There are still many challenges in current imitation learning, mainly including the following:

- The current imitation learning is for a specific task, and there is a lack of imitation learning methods that can be applied to multiple tasks;

- The current imitation learning algorithm is not optimal for expert data, and it is difficult to surpass expert data to achieve optimal results;

- The current imitation learning algorithm is mainly aimed at observation, and cannot combine multi-modal factors such as speech and natural language;

- Current imitation learning can find local optima, but often cannot find global optima.

Reference
--------

[1] Michael Bain and Claude Sammut. 1999. A framework for behavioural cloning. In *Machine Intelligence 15*. Oxford University Press, 103-129.

[2] Stéphane Ross, Geoffffrey Gordon, and Drew Bagnell. 2011. A reduction of imitation learning and structured prediction to no-regret online learning. In *Proceedings of the fourteenth international conference on artifificial intelligence and* *statistics*. JMLR Workshop and Conference Proceedings, 627-635.

[3] Florence, P. , Lynch, C. , Zeng, A. , Ramirez, O. , Wahid, A. , & Downs, L. , et al. (2021). Implicit behavioral cloning.

[4] Brian D Ziebart, Andrew L Maas, J Andrew Bagnell, and Anind K Dey. 2008. Maximum entropy inverse reinforcement learning.. In *Aaai*, Vol. 8. Chicago, IL, USA, 1433-1438.

[5] Chelsea Finn, Sergey Levine, and Pieter Abbeel. 2016. Guided cost learning: Deep inverse optimal control via policy optimization. In *International conference on machine learning*. PMLR, 49-58.

[6] Todd Hester, Matej Vecerik, Olivier Pietquin, Marc Lanctot, Tom Schaul, Bilal Piot, Dan Horgan, John Quan, Andrew Sendonaris, Gabriel Dulac-Arnold, Ian Osband, John Agapiou, Joel Z. Leibo, and Audrunas Gruslys. 2017. Deep Q learning from Demonstrations. *arXiv:1704.03732 [cs]* (Nov. 2017). http://arxiv.org/abs/1704.03732 arXiv: 1704.03732.

[7] Daniel Brown, Wonjoon Goo, Prabhat Nagarajan, and Scott Niekum. 2019. Extrapolating beyond suboptimal demonstrations via inverse reinforcement learning from observations. In *International Conference on Machine Learning*. PMLR, 783-792.

[8] Jonathan Ho and Stefano Ermon. 2016. Generative Adversarial Imitation Learning. In *Advances in Neural Information* *Processing Systems 29*, D. D. Lee, M. Sugiyama, U. V. Luxburg, I. Guyon, and R. Garnett (Eds.). Curran Associates, Inc., 4565-4573. http://papers.nips.cc/paper/6391-generative-adversarial-imitation-learning.pdf

[9] Yunzhu Li, Jiaming Song, and Stefano Ermon. 2017. InfoGAIL: Interpretable Imitation Learning from Visual Demonstrations. In *Advances in Neural Information Processing Systems 30*, I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (Eds.). Curran Associates, Inc., 3812-3822. http://papers.nips.cc/paper/6971-infogail-interpretable-imitation-learning-from-visual-demonstrations.pdf

[10] Yiming Ding, Carlos Florensa, Mariano Phielipp, and Pieter Abbeel. 2019. Goal-conditioned imitation learning. *arXiv* *preprint arXiv:1906.05838* (2019).

[11] Akira Kinose and Tadahiro Taniguchi. 2020. Integration of imitation learning using GAIL and reinforcement learning using task-achievement rewards via probabilistic graphical model. *Advanced Robotics* (June 2020), 1-13. https://doi.org/10.1080/01691864.2020.1778521

[12] Guoyu Zuo, Kexin Chen, Jiahao Lu, and Xiangsheng Huang. 2020. Deterministic generative adversarial imitation learning. *Neurocomputing* 388 (May 2020), 60-69. https://doi.org/10.1016/j.neucom.2020.01.016
