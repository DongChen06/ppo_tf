import argparse
from trainer import Trainer, Tester


def parse_args():
    parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                  'using Proximal Policy Optimizer'))
    parser.add_argument('-env_name', type=str, help='OpenAI Gym environment name', default='LunarLanderContinuous-v2')  # Pendulum-v0  LunarLanderContinuous-v2
    parser.add_argument('-n', '--num_episodes', type=int, help='Number of episodes to run',
                        default=2000)
    parser.add_argument('-g', '--gamma', type=float, help='Discount factor', default=0.995)
    parser.add_argument('-l', '--lam', type=float, help='Lambda for Generalized Advantage Estimation',
                        default=0.98)
    parser.add_argument('-k', '--kl_targ', type=float, help='D_KL target value',
                        default=0.003)
    parser.add_argument('-b', '--batch_size', type=int,help='Number of episodes per training batch',
                        default=20)
    parser.add_argument('-m', '--hid1_mult', type=int, help='Size of first hidden layer for value and policy NNs'
                             '(integer multiplier of observation dimension)', default=10)
    parser.add_argument('-v', '--policy_logvar', type=float,
                        help='Initial policy log-variance (natural log of variance)',
                        default=-1.0)
    parser.add_argument('-c', '--clipping_range', nargs=2, type=float,
                        help='Use clipping range objective in PPO instead of KL divergence penalty',
                        default=None)
    args = parser.parse_args()
    return args


def train_fn(env_name, num_episodes, gamma, lam, kl_targ, batch_size, hid1_mult, policy_logvar, clipping_range):
    train_fn = Trainer(env_name, batch_size, gamma, lam, kl_targ, num_episodes, hid1_mult, policy_logvar, clipping_range)
    train_fn.run()


def evaluate_fn(env_name):
    eval_fn = Tester(env_name)
    eval_fn.run()

if __name__ == "__main__":
    args = parse_args()
    is_training = True

    if is_training is True:
        train_fn(**vars(args))
    else:
        evaluate_fn(**vars(args))
