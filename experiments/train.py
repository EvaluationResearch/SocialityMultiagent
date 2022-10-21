import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import csv
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
from maddpg.trainer.maddpg import MADDPG3AgentTrainer
import tensorflow.contrib.layers as layers
import random
def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_tag", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=3, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="ddpg", help="policy for good agents")
    parser.add_argument("--bad-policy", type=str, default="ddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--adv-eps", type=float, default=1e-3, help="adversarial training rate")
    parser.add_argument("--adv-eps-s", type=float, default=1e-5, help="small adversarial training rate")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="XXX", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--load-name", type=str, default="",
                        help="name of which training state and model are loaded, leave blank to load seperately")
    parser.add_argument("--load-good", type=str, default="", help="which good policy to load")
    parser.add_argument("--load-bad", type=str, default="", help="which bad policy to load")
    # Evaluation
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/",
                        help="directory where plot data is saved")
    return parser.parse_args()
def parse_argsM():
    parserM = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parserM.add_argument("--scenario", type=str, default="simple_tag", help="name of the scenario script")
    parserM.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parserM.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parserM.add_argument("--num-adversaries", type=int, default=3, help="number of adversaries")
    parserM.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parserM.add_argument("--bad-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parserM.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parserM.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parserM.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parserM.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parserM.add_argument("--adv-eps", type=float, default=1e-3, help="adversarial training rate")
    parserM.add_argument("--adv-eps-s", type=float, default=1e-5, help="small adversarial training rate")
    # Checkpointing
    parserM.add_argument("--exp-name", type=str, default="XXX", help="name of the experiment")
    parserM.add_argument("--save-dir", type=str, default="/tmp/policy/",
                        help="directory in which training state and model should be saved")
    parserM.add_argument("--save-rate", type=int, default=1000,
                        help="save model once every time this many episodes are completed")
    parserM.add_argument("--load-name", type=str, default="",
                        help="name of which training state and model are loaded, leave blank to load seperately")
    parserM.add_argument("--load-good", type=str, default="", help="which good policy to load")
    parserM.add_argument("--load-bad", type=str, default="", help="which bad policy to load")
    # Evaluation
    parserM.add_argument("--test", action="store_true", default=False)
    parserM.add_argument("--restore", action="store_true", default=False)
    parserM.add_argument("--display", action="store_true", default=False)
    parserM.add_argument("--benchmark", action="store_true", default=False)
    parserM.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parserM.add_argument("--benchmark-dir", type=str, default="./benchmark_files/",
                        help="directory where benchmark data is saved")
    parserM.add_argument("--plots-dir", type=str, default="./learning_curves/",
                        help="directory where plot data is saved")
    return parserM.parse_args()
def parse_argsL():
    parserL = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parserL.add_argument("--scenario", type=str, default="simple_tag", help="name of the scenario script")
    parserL.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parserL.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parserL.add_argument("--num-adversaries", type=int, default=3, help="number of adversaries")
    parserL.add_argument("--good-policy", type=str, default="mmmaddpg", help="policy for good agents")
    parserL.add_argument("--bad-policy", type=str, default="mmmaddpg", help="policy of adversaries")
    # Core training parameters
    parserL.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parserL.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parserL.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parserL.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parserL.add_argument("--adv-eps", type=float, default=1e-3, help="adversarial training rate")
    parserL.add_argument("--adv-eps-s", type=float, default=1e-5, help="small adversarial training rate")
    # Checkpointing
    parserL.add_argument("--exp-name", type=str, default="XXX", help="name of the experiment")
    parserL.add_argument("--save-dir", type=str, default="/tmp/policy/",
                        help="directory in which training state and model should be saved")
    parserL.add_argument("--save-rate", type=int, default=1000,
                        help="save model once every time this many episodes are completed")
    parserL.add_argument("--load-name", type=str, default="",
                        help="name of which training state and model are loaded, leave blank to load seperately")
    parserL.add_argument("--load-good", type=str, default="", help="which good policy to load")
    parserL.add_argument("--load-bad", type=str, default="", help="which bad policy to load")
    # Evaluation
    parserL.add_argument("--test", action="store_true", default=False)
    parserL.add_argument("--restore", action="store_true", default=False)
    parserL.add_argument("--display", action="store_true", default=False)
    parserL.add_argument("--benchmark", action="store_true", default=False)
    parserL.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parserL.add_argument("--benchmark-dir", type=str, default="./benchmark_files/",
                        help="directory where benchmark data is saved")
    parserL.add_argument("--plots-dir", type=str, default="./learning_curves/",
                        help="directory where plot data is saved")
    return parserL.parse_args()
def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):

    trainers = []
    model = mlp_model
    trainer3 = MADDPG3AgentTrainer
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        if i==0:

            policy_name = "ddpg"
            print("{} predator agents :".format(i) + policy_name)
            trainers.append(trainer3(
                "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
                policy_name == 'ddpg', "ddpg", policy_name == 'mmmaddpg'))
        elif i==1:
            policy_name = "mmmaddpg"
            print("{} predator agents :".format(i) + policy_name)
            trainers.append(trainer3(
                "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
                policy_name == 'ddpg', "mmmaddpg", policy_name == 'mmmaddpg'))

        else:
            policy_name = "maddpg"
            print("{} predator agents :".format(i) + policy_name)
            trainers.append(trainer3(
                "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
                policy_name == 'ddpg', "maddpg", policy_name == 'mmmaddpg'))
    for i in range(num_adversaries, env.n):

        if i == 3 :
            policy_name = "maddpg"
            print("{} prey agents :".format(i) + policy_name)
            trainers.append(trainer3(
                "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
                policy_name == 'ddpg', "maddpg", policy_name == 'mmmaddpg'))
        elif i == 4:
            policy_name = "maddpg"
            print("{} prey agents :".format(i) + policy_name)
            trainers.append(trainer3(
                "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
                policy_name == 'ddpg', "maddpg", policy_name == 'mmmaddpg'))
        else:
            policy_name = "mmmaddpg"
            print("{} prey agents :".format(i) + policy_name)
            trainers.append(trainer3(
                "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
                policy_name == 'ddpg', "mmmaddpg", policy_name == 'mmmaddpg'))
    return trainers

def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        # print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        if arglist.test or arglist.display or arglist.restore or arglist.benchmark:
            if arglist.load_name == "":
                # load seperately
                bad_var_list = []
                for i in range(num_adversaries):
                    bad_var_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=trainers[i].scope)
                saver = tf.train.Saver(bad_var_list)
                U.load_state(arglist.load_bad, saver)

                good_var_list = []
                for i in range(num_adversaries, env.n):
                    good_var_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=trainers[i].scope)
                saver = tf.train.Saver(good_var_list)
                U.load_state(arglist.load_good, saver)
            else:
                print('Loading previous state from {}'.format(arglist.load_name))
                U.load_state(arglist.load_name)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        # adversary0_reward = 0
        adversary1_reward = 0
        adversary2_reward = 0
        adversary3_reward = 0
        agent0_reward = 0
        agent1_reward=0
        agent2_reward = 0
        agent3_reward = 0
        agent4_reward = 0
        agent5_reward = 0
        agent6_reward = 0
        agent7_reward = 0
        t_start = time.time()

        print('Starting iterations...')
        filename = 'data.txt'
        with open(filename, 'w') as f:
            while True:
                # get action
                action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]

                new_obs_n, rew_n, done_n, info_n = env.step(action_n)
                #If you need a random agent...
                # action_n[0] = [random.random(), random.random(), random.random(), random.random(), random.random()]
                # action_n[1] = [random.random(), random.random(), random.random(), random.random(), random.random()]
                action_n[2] = [random.random(), random.random(), random.random(), random.random(), random.random()]
                # action_n[3] = [random.random(), random.random(), random.random(), random.random(), random.random()]
                action_n[4] = [random.random(), random.random(), random.random(), random.random(), random.random()]
                # action_n[5] = [random.random(), random.random(), random.random(), random.random(), random.random()]
                # action_n[6] = [random.random(), random.random(), random.random(), random.random(), random.random()]

                agent0_reward += rew_n[0]
                agent1_reward += rew_n[1]
                agent2_reward += rew_n[2]
                agent3_reward += rew_n[3]
                agent4_reward += rew_n[4]
                agent5_reward += rew_n[5]
                # agent6_reward += rew_n[6]
                # agent7_reward += rew_n[7]
                # agent1_reward += rew_n[2]
                # agent2_reward += rew_n[3]
                # agent1_reward += rew_n[2]
                # f.write("%s %s %s %s"%(str(rew_n[0]),str(rew_n[1]),str(rew_n[2]),str(rew_n[3])))
                # f.write("\n")
                episode_step += 1
                done = all(done_n)
                terminal = (episode_step >= arglist.max_episode_len)
                # collect experience
                for i, agent in enumerate(trainers):
                    agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
                obs_n = new_obs_n

                for i, rew in enumerate(rew_n):
                    episode_rewards[-1] += rew
                    agent_rewards[i][-1] += rew

                if done or terminal:
                    obs_n = env.reset()
                    episode_step = 0
                    episode_rewards.append(0)
                    for a in agent_rewards:
                        a.append(0)
                    agent_info.append([[]])

                # increment global step counter
                train_step += 1

                # for benchmarking learned policies
                if arglist.benchmark:
                    for i, info in enumerate(info_n):
                        agent_info[-1][i].append(info_n['n'])
                    if train_step > arglist.benchmark_iters and (done or terminal):
                        file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                        print('Finished benchmarking, now saving...')
                        with open(file_name, 'wb') as fp:
                            pickle.dump(agent_info[:-1], fp)
                        break
                    continue

                # for displaying learned policies
                if arglist.display:
                    time.sleep(0.1)
                    env.render()
                    continue

                # update all trainers, if not in display or benchmark mode
                loss = None
                for agent in trainers:
                    agent.preupdate()
                for agent in trainers:
                    loss = agent.update(trainers, train_step)

                # save model, display training output
                if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                    U.save_state(arglist.save_dir, saver=saver)
                    # print statement depends on whether or not there are adversaries
                    b=arglist.save_rate
                    a=episode_rewards[-arglist.save_rate:]
                    if num_adversaries == 0:
                        print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                            train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                            round(time.time() - t_start, 3)))
                    else:
                        print(
                            "steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                                train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                                [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards],
                                round(time.time() - t_start, 3)))
                    t_start = time.time()
                    # a=[]
                    # a.append([np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards])
                    # b=a[0][0]
                    #
                    # with open('data.csv', 'w') as f:
                    #     writer = csv.writer(f)
                    #     writer.writerow([train_step,len(episode_rewards),a[0][0]+a[0][1],a[0][2]+a[0][3]])
                    # Keep track of final episode reward
                    final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                    for rew in agent_rewards:
                        final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

                # saves final episode reward for plotting training curve later
                if len(episode_rewards) > arglist.num_episodes:
                    rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                    with open(rew_file_name, 'wb') as fp:
                        pickle.dump(final_ep_rewards, fp)
                    agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                    with open(agrew_file_name, 'wb') as fp:
                        pickle.dump(final_ep_ag_rewards, fp)
                    Tadversary_reward = agent0_reward + agent1_reward+ agent2_reward
                    Tgood_reward = agent3_reward+ agent4_reward+ agent5_reward
                    # Tadversary_reward=agent0_reward+agent1_reward+agent2_reward
                    # Tgood_reward =agent3_reward+agent4_reward+agent5_reward
                    print('...total Tadversary_reward is %f' % (Tadversary_reward))
                    print('...total adversary0_reward is %f' % (agent0_reward))
                    print('...total adversary1_reward is %f' % (agent1_reward))
                    print('...total adversary2_reward is %f' % (agent2_reward))


                    print('...total good_reward is %f' % (Tgood_reward))
                    print('...total good0_reward is %f' % (agent3_reward))
                    print('...total good1_reward is %f' % (agent4_reward))
                    print('...total good2_reward is %f' % (agent5_reward))
                    # print('...total good1_reward is %f' % (agent5_reward))
                    Tadversary_reward=int(Tadversary_reward//1000)
                    agent0_reward = int(agent0_reward // 1000)
                    agent1_reward = int(agent1_reward // 1000)
                    Tgood_reward = int(Tgood_reward // 1000)
                    agent2_reward = int(agent2_reward // 1000)
                    agent3_reward = int(agent3_reward // 1000)
                    agent4_reward = int(agent4_reward // 1000)
                    agent5_reward = int(agent5_reward // 1000)
                    # print('&', Tadversary_reward,'k', '&', agent0_reward,'k','&',agent1_reward,'k', '&',Tgood_reward,'k', '&', agent2_reward,'k', '&',agent3_reward,'k')
                    print('&', Tadversary_reward,'k', '&', agent0_reward,'k','&',agent1_reward,'k','&', agent2_reward,'k', '&',Tgood_reward,'k', '&', agent3_reward,'k', '&',agent4_reward,'k','&', agent5_reward,'k',)
                    # print('&', format(Tadversary_reward, '.1f'), '&',format(agent0_reward, '.1f'),'&',format(agent1_reward, '.1f'), '&',format(agent2_reward, '.1f'), '&', format(Tgood_reward, '.1f'),'&', format(agent3_reward, '.1f'),'&', format(agent4_reward, '.1f'),'&', format(agent5_reward, '.1f'))


                    #
                    # print('...total good3_reward is %f' % (agent3_reward))

                    # print('...total agent2_reward is %f' % (agent2_reward))
                    print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                    f.close()
                    break




if __name__ == '__main__':


    arglist = parse_args()

    train(arglist)
