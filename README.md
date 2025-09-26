## Multi-agent Guided Policy Search (MA-GPS)

This repository contains the implementation of Guided Policy Search for Multi-Agent General-Sum Dynamic Games. We propose two different environments for testing our approach: one is three-unicycle platooning, and the other is six-player basketball training. If you are interested in learning more about this work, please refer to the following paper: Guided Policy Search for Multi-Agent General-Sum Dynamic Games, J. Li*, G. Qu*. J. Choi, S. Sojoudi, C. Tomlin.

Our repo defines general-sum dynamic games using gym env API. You can find general-sum dynamic games examples in /MAGPS/MARL_gym_envs/

# Introduction

Reinforcement learning (RL) has been widely applied to solving challenging decision-making problems such as the game of Go (Silver et al., 2017) and legged locomotion (Hwangbo et al., 2019). However, most of the reported successes in RL focus on single-agent models. Many problems where multiple agents must coordinate efficiently under decentralized decision-making are often overlooked (Gronauer and Diepold, 2022). For example, a group of robots learning to operate efficiently in a warehouse must account for what others want to achieve and how they are learning (Wang et al., 2022). Ensuring that these learning agents coordinate well and efficiently optimize their individual objectives—which are often misaligned—is a challenging problem (Wong et al., 2023). This challenge can be framed as the problem of learning a Nash equilibrium of a non-cooperative general-sum game (Zhang et al., 2021). As suggested in prior works such as Mazumdar et al. (2020), though model-free multi-agent policy gradient methods are capable of learning complex policies for high dimensional systems, they are usually unstable and can fall into limit cycles, failing to converge to a Nash equilibrium.

Classical approaches to solving non-cooperative games typically rely on model-based methods (Bas ̧ar and Olsder, 1998). However, such problems are computationally challenging due to the presence of mis- aligned individual objectives and complex dynamics. While smooth system dynamics may mitigate some computational difficulties, identifying Nash equilibria still poses significant challenges due to the intricatecoupling among the players’ objectives (Conitzer and Sandholm, 2008; Dreves et al., 2011). In Fridovich- Keil et al. (2020) and Laine et al. (2023), the authors use the local linear quadratic models of the game and KKT conditions to reliably compute local Nash equilibrium policies via algorithms inspired by Newton’s method. However, as the game becomes more complex—due to factors such as an increased number of players, nonlinear dynamics, or complex objectives—the computation can slow down significantly and the solution may converge to an inferior policy (Kossioris et al., 2008).

In addition, both of our methods can be computed in real-time to certify if a neighboring set of the current state is within the ground truth reach-avoid set. The computational complexity of evaluating our two certifications scales **polynomially** with respect to the state dimensions. Our method can also be used offline for a comprehensive certification, i.e., certifying if a large set of states is within the ground truth reach-avoid set.

The main idea of our work is to combine the strengths of model-free policy gradient methods and model- based algorithms to overcome their limitations in computing Nash equilibria of non-cooperative games. Our primary strategy is to use model-based approximate policies to guide the model-free policy search through reward regularization, which stabilizes the policy gradient method and leads to convergence toward superior policies. While model-based guidance and reward regularization have been extensively studied in single- agent RL settings (Levine and Koltun, 2013; Liu et al., 2020), extending these approaches to multi-agent settings for computing Nash equilibria of general-sum games is non-trivial. The primary challenge arises from misaligned objectives among agents (Mazumdar et al., 2020; Wong et al., 2023), which can lead to instability in learning dynamics. Moreover, existing methods for computing model-based Nash equilibria (Di and Lamperski, 2019; Fridovich-Keil et al., 2020; Laine et al., 2023) are computationally intensive and not fast enough for real-time applications, a crucial requirement for the efficient training of MARL policies.

To address these gaps, we introduce the first method that leverages real-time computed, model-based approximate game solution guidance for multi-agent reinforcement learning (MARL) in non-cooperative games. Our approach facilitates stable and efficient policy learning of Nash equilibria in complex general- sum games. We theoretically characterize the stability and local convergence of our guided policy gradient dynamics in linear-quadratic games, and we also empirically validate the high learning efficiency of our method across various nonlinear general-sum games.

# Implementation details and installation guide:

Our implementation builds upon the deep RL infrastructure of [Tianshou](https://github.com/thu-ml/tianshou) (version 0.5.1).  

We recommend Python version 3.12. 

Install instruction:

1. git clone the repo

2. cd to the root location of this repo, where you should be able to see the "setup.py". Note that if you use MacOS, then pytorch 2.4.0 is not available, and therefore you have to first change the line 22 of setup.py from "pytorch==2.4.0" to "pytorch==2.2.2", and then do the step 3. (However, Pytorch==2.4.0 is available for Ubuntu systems. So, if you use Ubuntu, then you can directly go to step 3. )

3. run in terminal: pip install -e .

4. run in terminal: conda install -c conda-forge ffmpeg

Brief tutorial: we use `experiment_script/run_magps.py` to learn our policy. We visualize the results  in `experiment_script/eval_magps_three_unicycle.ipynb` and `experiment_script/eval_magps_six_basketball.ipynb`. 

# Some sample training scripts:

For three unicycle platooning

> python run_magps.py --task Three_Unicycle_Game-v0 --critic-net 512 512 512 --actor-net 512 512 512 --epoch 15 --total-episodes 160 --gamma 0.99 --behavior-loss-weight 0.1 --batch-size 2048

For six basketball players in training:

> python run_magps.py --task basketball-v0 --critic-net 512 512 512 --actor-net 512 512 512 --epoch 15 --total-episodes 160 --gamma 0.99 --behavior-loss-weight 0.1 --batch-size 2048


# Future directions:

Our method stabilizes the limit cycle behavior of policy gradient in vanilla LQ games, and its deep RL extension efficiently learns complex Nash equilibrium strategies. In future work, we aim to extend this framework to multi-agent reinforcement learning (MARL) with partial observability and investigate the role of policy guidance in facilitating efficient learning. Additionally, we plan to explore MARL under safety constraints to further enhance its applicability in real-world scenarios. 
