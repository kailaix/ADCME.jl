# Reinforcement Learning Basics: Q-learning and SARSA 

This note gives a short introduction to Q-learning and SARSA in reinforcement learning. 

## Reinforcement Learning

Reinforcement learning aims at making optimal decisions using experiences. In reinforcement learning, an **agent** interacts with an **environment**. There are three important concepts in reinforcement learning: **states**, **actions**, and **rewards**. An agent in a certain state takes an action, which results in a reward from the environment and a change of states. In this note, we assume that the states and actions are discrete, and let $\mathcal{S}$ and $\mathcal{A}$ denotes the set of states and actions, respectively. 

We first define some important concepts in reinforcement learning:

- Policy. A policy is a function $\pi: \mathcal{S}\rightarrow \mathcal{A}$ that defines the agent's action at a given state. 
- Reward. A reward is a function $R: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$ whose value is rewarded to an agent for performing certain actions at a given state. Particularly, the goal of an agent is to maximum the discounted reward
  
$$\min_\pi\ R = \sum_{t=0}^\infty \gamma^t R(s_{t+1}, a_{t+1}) \tag{1}$$

where $\gamma\in (0,1)$ is a discounted factor, and $(s_t, a_t)$ are a sequence of states and actions following a policy $\pi$. 

Specifically, in this note we assume that both the policy and the reward function are time-independent and deterministic.  

We can now define the quality function (Q-function) for a given policy 

$$Q^\pi(s_t, a_t) = \mathbf{E}(R(s_t, a_t) + \gamma R(s_{t+1}, a_{t+1}) + \gamma^2 R(s_{t+2}, a_{t+2})  + \ldots | s_t, a_t)$$

here $\pi$ is a given policy

It can be shown that the solution to Equation 1 is given by the policy $\pi$ that satisfies

$$\pi(s) = \max_a\ Q^\pi(s, a)$$

## Q-learning and SARSA 

Both Q-learning and SARSA learn the Q-function iteratively. We denote the everchanging Q-function in the iterative process as $Q(s,a)$ (no superscript referring to any policy). When $|\mathcal{S}|<\infty, |\mathcal{A}|<\infty$, $Q(s,a)$ can be tabulated as a $|\mathcal{S}|\times |\mathcal{A}|$ table. 

A powerful technique in reinforcment learning is the epsilon-greedy algorithm, which strikes a balance between exploration and exploitation of the state space. To describe the espilon-greedy algorithm, we introduce the **stochastic policy** $\pi_\epsilon$ given a Q-function:

$$\pi_\epsilon(s) = \begin{cases}a' & \text{w.p.}\ \epsilon \\ \arg\max_a Q(s, a) &\text{w.p.}\ 1-\epsilon\end{cases}$$

Here $a'$ is a random variable whose values are in $\mathcal{A}$ (e.g., a uniform random variable over $\mathcal{A}$). 

Then the Q-learning update formula can be expressed as 

$$Q(s,a) = (1-\alpha)Q(s,a) + \alpha \left(R(s,a) + \gamma\max_{a'} Q(s', a')\right)$$

The SARSA update formula can be expressed as 

$$Q(s,a) \gets (1-\alpha)Q(s,a) + \alpha \left(R(s,a) + \gamma Q
(s', a')\right),\ a' = \pi_\epsilon(s')$$

In both cases, $s'$ is the subsequent state given the last action $a$ at state $s$, and $a = \pi_\epsilon(s)$. 

The subtle difference between Q-learning and SARSA is **how you select your next best action**, either max or mean.

To extract the optimal deterministic policy $\pi$ from $\pi_\epsilon$, we only need to define

$$\pi(s) := \arg\max_a Q(s,a)$$

## Examples

We use OpenAI gym to perform numerical experiments. We reimplemented the Q-learning algorithm from [this post](https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/) in Julia. 

- Q-learning: [code](https://github.com/kailaix/ADCME.jl/blob/master/docs/src/assets/Codes/ML/qlearning.jl)
- SARSA: [code](https://github.com/kailaix/ADCME.jl/blob/master/docs/src/assets/Codes/ML/salsa.jl)

To run the scripts, you need to install the dependencies via 

```julia
using ADCME
PIP = get_pip()
run(`$PIP install cmake 'gym[atari]'`)
```


