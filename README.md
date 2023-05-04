You can access the code and data at SocialityMultiagent/experiments/Sociality/.
  1) The results for 2v2 and 3v3 games are 2.csv and 3.csv.
  2) SE.csv shows the result for selfish predators vs egalitarian prey.
  3) The outcome for altruistic prey is Aprey.csv.
  4) The outcome for the altruistic predator is A.csv.

We initially explored the values of sociality in {0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0 } for several combinations of teams with two agents. What we found is that the average results are mostly monotonic (run 7Socalities.py). The data is shown in 7Soci.csv.

This is the code for implementing the MADDPG algorithm presented in the paper:
[Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf).
It is configured to be run in conjunction with environments from the
[Multi-Agent Particle Environments (MPE)](https://github.com/openai/multiagent-particle-envs).

##1.  Installation

- To install, `cd` into the root directory and type `pip install -e .`

- Known dependencies: Python (3.5.4), OpenAI gym (0.10.5), tensorflow (1.8.0), numpy (1.14.5)

- Ensure that `multiagent-particle-envs` has been added to your `PYTHONPATH` (e.g. in `~/.bashrc` or `~/.bash_profile`).

##2.  run the code to get the data

- We have implement the code with different agent (MADDPG, DDPG, mmmaddpg, and random) in  `train.py`, you could change the policy name in get_trainers to 
get 3 different agents. For random agent, " action_n[*] = [random.random(), random.random(), random.random(), random.random(), random.random()]" will let you get a random agent.

- For the  selfish, egalitarian and altruistic agent, please change the parameter "al" in adversary_rewardA and
agent_rewardA which marked "#change this!!!!!!!!!!!!!!" in SocialityMultiagent\multiagent\scenarios\simple_tag.py
- To run the , `cd` into the `experiments` directory and run `train.py`:

``python train.py --scenario simple_tag``

Then you will get the reward of each agent, and after run the different combination, you wil get the table 
13 and table 14 list in appendix. In our setting, if things go well, these experiments can be finished in 
around three weeks.


##3.  To get the result in paper, please run the code in SocialityMultiagent\experiments\sociality

- Please run fig2pred.py and fig2prey.py which will plot the fig2 in paper.
- ranktestMDLR.py  will plot the fig3 in paper.
- heatbot.py will plot fig 4 and fig 5.
- heat2.py and heat3.py are for fig 6.
- heatbotMDLR.py refers to fig 7 and fig 9(in appendix)
- NemenyiTest.py is for fig 8











