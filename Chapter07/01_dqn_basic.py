#!/usr/bin/env python3
import gym
import ptan
import argparse

import torch
import torch.optim as optim

from tensorboardX import SummaryWriter

from lib import dqn_model, common


if __name__ == "__main__":
    params = common.HYPERPARAMS['pong']
#    params['epsilon_frames'] = 200000
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = gym.make(params['env_name'])
    env = ptan.common.wrappers.wrap_dqn(env)

    writer = SummaryWriter(comment="-" + params['run_name'] + "-basic")
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)

    tgt_net = ptan.agent.TargetNet(net) # Target network (copy of net synchronized from time to time)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start']) # e-greedy selectoy of actions
    epsilon_tracker = common.EpsilonTracker(selector, params) # schedules epsilon according to the current frame number
    agent = ptan.agent.DQNAgent(net, selector, device=device) # agent class with Q-network and selector

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=1) # generates touples from the environment in the form (s, a, r, s')
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params['replay_size']) # buffer of experiences for experience replay
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

    frame_idx = 0

    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += 1
            buffer.populate(1) # put 1 sample in the experience buffer
            epsilon_tracker.frame(frame_idx) # set epsilon decay for this frame

            new_rewards = exp_source.pop_total_rewards() # check for finished episode and monitor their total reward
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    break

            if len(buffer) < params['replay_initial']:
                continue

            optimizer.zero_grad()
            batch = buffer.sample(params['batch_size']) # get batch from experience buffer
            loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params['gamma'], device=device) # compute loss
            loss_v.backward() # compute loss derivative
            optimizer.step() # optimization step

            if frame_idx % params['target_net_sync'] == 0:
                tgt_net.sync() # synchronize target network
