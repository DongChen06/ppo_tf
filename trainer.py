import gym
import numpy as np
from agents.actor_function import ActorNetwork
from agents.value_function import CriticNetwork
import scipy.signal
from common.utils import Logger, Scaler
from datetime import datetime
import tensorflow as tf


class Trainer:
    def __init__(self, env_name, batch_size, gamma, lam, kl_targ, num_episodes, hid1_mult, policy_logvar, clipping_range):
        """
        Args:
            env_name: OpenAI Gym environment name, e.g. 'Hopper-v1'
            num_episodes: maximum number of episodes to run
            gamma: reward discount factor (float)
            lam: lambda from Generalized Advantage Estimate
            kl_targ: D_KL target for policy update [D_KL(pi_old || pi_new)
            batch_size: number of episodes per policy training batch
            hid1_mult: hid1 size for policy and value_f (mutliplier of obs dimension)
            policy_logvar: natural log of initial policy variance
        """
        self.env_name = env_name
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam
        self.kl_targ = kl_targ
        self.num_episodes = num_episodes
        self.hid1_mult = hid1_mult
        self.policy_logvar = policy_logvar
        self.clipping_range = clipping_range
        self.env = gym.make(self.env_name)

    def run_episode(self, policy, scaler, rendering=False):
        """ Run single episode with option to animate

        Args:
            policy: policy object with sample() method
            scaler: scaler object, used to scale/offset each observation dimension
                to a similar range
            rendering: boolean, True uses env.render() method to animate episode

        Returns: 4-tuple of NumPy arrays
            observes: shape = (episode len, obs_dim)
            actions: shape = (episode len, act_dim)
            rewards: shape = (episode len,)
            unscaled_obs: useful for training scaler, shape = (episode len, obs_dim)
        """
        obs = self.env.reset()
        observes, actions, rewards, unscaled_obs = [], [], [], []
        done = False
        step = 0.0
        scale, offset = scaler.get()
        scale[-1] = 1.0  # don't scale time step feature
        offset[-1] = 0.0  # don't offset time step feature
        while not done:
            if rendering:
                self.env.render()
            obs = obs.astype(np.float32).reshape((1, -1))
            obs = np.append(obs, [[step]], axis=1)  # add time step feature
            unscaled_obs.append(obs)
            obs = (obs - offset) * scale  # center and scale observations
            observes.append(obs)
            action = policy.forward(obs).reshape((1, -1)).astype(np.float32)
            actions.append(action)
            obs, reward, done, _ = self.env.step(np.squeeze(action, axis=0))
            if not isinstance(reward, float):
                reward = float(reward)
            rewards.append(reward)
            step += 1e-3  # increment time step feature

        return (np.concatenate(observes), np.concatenate(actions),
                np.array(rewards, dtype=np.float64), np.concatenate(unscaled_obs))

    def run_policy(self, policy, scaler, logger, episodes):
        """ Run policy and collect data for a minimum of min_steps and min_episodes

        Args:
            policy: policy object with sample() method
            scaler: scaler object, used to scale/offset each observation dimension
                to a similar range
            logger: logger object, used to save stats from episodes
            episodes: total episodes to run

        Returns: list of trajectory dictionaries, list length = number of episodes
            'observes' : NumPy array of states from episode
            'actions' : NumPy array of actions from episode
            'rewards' : NumPy array of (un-discounted) rewards from episode
            'unscaled_obs' : NumPy array of (un-discounted) rewards from episode
        """
        total_steps = 0
        trajectories = []
        for e in range(episodes):
            observes, actions, rewards, unscaled_obs = self.run_episode(policy, scaler)
            total_steps += observes.shape[0]
            trajectory = {'observes': observes,
                          'actions': actions,
                          'rewards': rewards,
                          'unscaled_obs': unscaled_obs}
            trajectories.append(trajectory)
        unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
        scaler.update(unscaled)  # update running statistics for scaling observations
        logger.log({'_MeanReward': np.mean([t['rewards'].sum() for t in trajectories]),
                    'Steps': total_steps})

        return trajectories

    def run(self,  ):
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        obs_dim += 1  # add 1 to obs dimension for time step feature (see run_episode())
        now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")  # create unique directories
        logger = Logger(logname=self.env_name, now=now)
        scaler = Scaler(obs_dim)

        # initialize tensorflow session
        tf.reset_default_graph()
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=1,
                                inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        actor_network = ActorNetwork(self.sess, obs_dim, act_dim, self.kl_targ, self.hid1_mult, self.policy_logvar,
                              self.clipping_range)
        critic_network = CriticNetwork(self.sess, obs_dim, self.hid1_mult)

        # initialize policy and value tensorflow graph
        actor_network.build_graph()
        critic_network.build_graph()
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

        # run a few episodes of untrained policy to initialize scaler:
        self.run_policy(actor_network, scaler, logger, episodes=5)
        episode = 0
        while episode < self.num_episodes:
            trajectories = self.run_policy(actor_network, scaler, logger, episodes=self.batch_size)
            episode += len(trajectories)
            self.add_value(trajectories, critic_network)  # add estimated values to episodes
            self.add_disc_sum_rew(trajectories)  # calculated discounted sum of Rs
            self.add_gae(trajectories)  # calculate advantage

            # concatenate all episodes into single NumPy arrays
            observes = np.concatenate([t['observes'] for t in trajectories])
            actions = np.concatenate([t['actions'] for t in trajectories])
            disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
            advantages = np.concatenate([t['advantages'] for t in trajectories])
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

            # add various stats to training log:
            self.log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode)
            # update policy
            actor_network.backward(observes, actions, advantages, logger)
            # update value function
            critic_network.backward(observes, disc_sum_rew, logger)
            # write logger results to file and stdout
            logger.write(display=True)
        logger.close()
        actor_network.close_sess()
        critic_network.close_sess()

    def add_disc_sum_rew(self, trajectories):
        """ Adds discounted sum of rewards to all time steps of all trajectories

        Args:
            trajectories: as returned by run_policy()

        Returns:
            None (mutates trajectories dictionary to add 'disc_sum_rew')
        """
        for trajectory in trajectories:
            if self.gamma < 0.999:  # don't scale for gamma ~= 1
                rewards = trajectory['rewards'] * (1 - self.gamma)
            else:
                rewards = trajectory['rewards']
            # calculate discounted forward sum of a sequence at each point
            disc_sum_rew = scipy.signal.lfilter([1.0], [1.0, -self.gamma], rewards[::-1])[::-1]
            trajectory['disc_sum_rew'] = disc_sum_rew

    def add_value(self, trajectories, val_func):
        """ Adds estimated value to all time steps of all trajectories

        Args:
            trajectories: as returned by run_policy()
            val_func: object with predict() method, takes observations
                and returns predicted state value

        Returns:
            None (mutates trajectories dictionary to add 'values')
        """
        for trajectory in trajectories:
            observes = trajectory['observes']
            values = val_func.predict(observes)
            trajectory['values'] = values

    def add_gae(self, trajectories):
        """ Add generalized advantage estimator.
        https://arxiv.org/pdf/1506.02438.pdf

        Args:
            trajectories: as returned by run_policy(), must include 'values'
                key from add_value().
            gamma: reward discount
            lam: lambda (see paper).
                lam=0 : use TD residuals
                lam=1 : A =  Sum Discounted Rewards - V_hat(s)

        Returns:
            None (mutates trajectories dictionary to add 'advantages')
        """
        for trajectory in trajectories:
            if self.gamma < 0.999:  # don't scale for gamma ~= 1
                rewards = trajectory['rewards'] * (1 - self.gamma)
            else:
                rewards = trajectory['rewards']
            values = trajectory['values']
            # temporal differences
            tds = rewards - values + np.append(values[1:] * self.gamma, 0)
            # calculate discounted forward sum of a sequence at each point
            advantages = scipy.signal.lfilter([1.0], [1.0, -self.gamma * self.lam], tds[::-1])[::-1]
            trajectory['advantages'] = advantages

    def log_batch_stats(self, observes, actions, advantages, disc_sum_rew, logger, episode):
        """ Log various batch statistics """
        logger.log({'_mean_obs': np.mean(observes),
                    '_min_obs': np.min(observes),
                    '_max_obs': np.max(observes),
                    '_std_obs': np.mean(np.var(observes, axis=0)),
                    '_mean_act': np.mean(actions),
                    '_min_act': np.min(actions),
                    '_max_act': np.max(actions),
                    '_std_act': np.mean(np.var(actions, axis=0)),
                    '_mean_adv': np.mean(advantages),
                    '_min_adv': np.min(advantages),
                    '_max_adv': np.max(advantages),
                    '_std_adv': np.var(advantages),
                    '_mean_discrew': np.mean(disc_sum_rew),
                    '_min_discrew': np.min(disc_sum_rew),
                    '_max_discrew': np.max(disc_sum_rew),
                    '_std_discrew': np.var(disc_sum_rew),
                    '_Episode': episode
                    })


class Tester:
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(self.env_name)

    def run(self):
        pass