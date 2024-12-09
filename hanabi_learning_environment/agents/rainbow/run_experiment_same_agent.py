# coding=utf-8
# Copyright 2018 The Dopamine Authors and Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
#
# This file is a fork of the original Dopamine code incorporating changes for
# the multiplayer setting and the Hanabi Learning Environment.
#
"""Run methods for training a DQN agent on Atari.

Methods in this module are usually referenced by |train.py|.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy  # For deep copy of the training agent
import time

from third_party.dopamine import checkpointer
from third_party.dopamine import iteration_statistics
import dqn_agent
import gin.tf
from hanabi_learning_environment import rl_env
import numpy as np
import rainbow_agent
import tensorflow as tf

LENIENT_SCORE = False


class ObservationStacker(object):
  """Class for stacking agent observations."""

  def __init__(self, history_size, observation_size, num_players):
    """Initializer for observation stacker.

    Args:
      history_size: int, number of time steps to stack.
      observation_size: int, size of observation vector on one time step.
      num_players: int, number of players.
    """
    self._history_size = history_size
    self._observation_size = observation_size
    self._num_players = num_players
    self._obs_stacks = list()
    for _ in range(0, self._num_players):
      self._obs_stacks.append(np.zeros(self._observation_size *
                                       self._history_size))

  def add_observation(self, observation, current_player):
    """Adds observation for the current player.

    Args:
      observation: observation vector for current player.
      current_player: int, current player id.
    """
    self._obs_stacks[current_player] = np.roll(self._obs_stacks[current_player],
                                               -self._observation_size)
    self._obs_stacks[current_player][(self._history_size - 1) *
                                     self._observation_size:] = observation

  def get_observation_stack(self, current_player):
    """Returns the stacked observation for current player.

    Args:
      current_player: int, current player id.
    """

    return self._obs_stacks[current_player]

  def reset_stack(self):
    """Resets the observation stacks to all zero."""

    for i in range(0, self._num_players):
      self._obs_stacks[i].fill(0.0)

  @property
  def history_size(self):
    """Returns number of steps to stack."""
    return self._history_size

  def observation_size(self):
    """Returns the size of the observation vector after history stacking."""
    return self._observation_size * self._history_size


def load_gin_configs(gin_files, gin_bindings):
  """Loads gin configuration files.

  Args:
    gin_files: A list of paths to the gin configuration files for this
      experiment.
    gin_bindings: List of gin parameter bindings to override the values in the
      config files.
  """
  gin.parse_config_files_and_bindings(gin_files,
                                      bindings=gin_bindings,
                                      skip_unknown=False)


@gin.configurable
def create_environment(game_type='Hanabi-Small', num_players=2):
  """Creates the Hanabi environment.

  Args:
    game_type: Type of game to play. Currently the following are supported:
      Hanabi-Full: Regular game.
      Hanabi-Small: The small version of Hanabi, with 2 cards and 2 colours.
    num_players: Int, number of players to play this game.

  Returns:
    A Hanabi environment.
  """
  return rl_env.make(
      environment_name=game_type, num_players=num_players, pyhanabi_path=None)


@gin.configurable
def create_obs_stacker(environment, history_size=4):
  """Creates an observation stacker.

  Args:
    environment: environment object.
    history_size: int, number of steps to stack.

  Returns:
    An observation stacker object.
  """

  return ObservationStacker(history_size,
                            environment.vectorized_observation_shape()[0],
                            environment.players)


@gin.configurable
def create_agent(environment, obs_stacker, agent_type='DQN'):
  """Creates the Hanabi agent.

  Args:
    environment: The environment.
    obs_stacker: Observation stacker object.
    agent_type: str, type of agent to construct.

  Returns:
    An agent for playing Hanabi.

  Raises:
    ValueError: if an unknown agent type is requested.
  """
  if agent_type == 'DQN':
    return dqn_agent.DQNAgent(observation_size=obs_stacker.observation_size(),
                              num_actions=environment.num_moves(),
                              num_players=environment.players)
  elif agent_type == 'Rainbow':
    return rainbow_agent.RainbowAgent(
        observation_size=obs_stacker.observation_size(),
        num_actions=environment.num_moves(),
        num_players=environment.players)
  else:
    raise ValueError('Expected valid agent_type, got {}'.format(agent_type))


def initialize_checkpointing(agent, experiment_logger, checkpoint_dir,
                             checkpoint_file_prefix='ckpt'):
  """Reloads the latest checkpoint if it exists.

  The following steps will be taken:
   - This method will first create a Checkpointer object, which will be used in
     the method and then returned to the caller for later use.
   - It will then call checkpointer.get_latest_checkpoint_number to determine
     whether there is a valid checkpoint in checkpoint_dir, and what is the
     largest file number.
   - If a valid checkpoint file is found, it will load the bundled data from
     this file and will pass it to the agent for it to reload its data.
   - If the agent is able to successfully unbundle, this method will verify that
     the unbundled data contains the keys, 'logs' and 'current_iteration'. It
     will then load the Logger's data from the bundle, and will return the
     iteration number keyed by 'current_iteration' as one of the return values
     (along with the Checkpointer object).

  Args:
    agent: The agent that will unbundle the checkpoint from checkpoint_dir.
    experiment_logger: The Logger object that will be loaded from the
      checkpoint.
    checkpoint_dir: str, the directory containing the checkpoints.
    checkpoint_file_prefix: str, the checkpoint file prefix.

  Returns:
    start_iteration: int, The iteration number to start the experiment from.
    experiment_checkpointer: The experiment checkpointer.
  """
  experiment_checkpointer = checkpointer.Checkpointer(
      checkpoint_dir, checkpoint_file_prefix)

  start_iteration = 0

  # Check if checkpoint exists. Note that the existence of checkpoint 0 means
  # that we have finished iteration 0 (so we will start from iteration 1).
  latest_checkpoint_version = checkpointer.get_latest_checkpoint_number(
      checkpoint_dir)
  if latest_checkpoint_version >= 0:
    dqn_dictionary = experiment_checkpointer.load_checkpoint(
        latest_checkpoint_version)
    if agent.unbundle(
        checkpoint_dir, latest_checkpoint_version, dqn_dictionary):
      assert 'logs' in dqn_dictionary
      assert 'current_iteration' in dqn_dictionary
      experiment_logger.data = dqn_dictionary['logs']
      start_iteration = dqn_dictionary['current_iteration'] + 1
      tf.logging.info('Reloaded checkpoint and will start from iteration %d',
                      start_iteration)

  return start_iteration, experiment_checkpointer


def format_legal_moves(legal_moves, action_dim):
  """Returns formatted legal moves.

  This function takes a list of actions and converts it into a fixed size vector
  of size action_dim. If an action is legal, its position is set to 0 and -Inf
  otherwise.
  Ex: legal_moves = [0, 1, 3], action_dim = 5
      returns [0, 0, -Inf, 0, -Inf]

  Args:
    legal_moves: list of legal actions.
    action_dim: int, number of actions.

  Returns:
    a vector of size action_dim.
  """
  new_legal_moves = np.full(action_dim, -float('inf'))
  if legal_moves:
    new_legal_moves[legal_moves] = 0
  return new_legal_moves


def parse_observations(observations, num_actions, obs_stacker):
  """Deconstructs the rich observation data into relevant components.

  Args:
    observations: dict, containing full observations.
    num_actions: int, The number of available actions.
    obs_stacker: Observation stacker object.

  Returns:
    current_player: int, Whose turn it is.
    legal_moves: `np.array` of floats, of length num_actions, whose elements
      are -inf for indices corresponding to illegal moves and 0, for those
      corresponding to legal moves.
    observation_vector: Vectorized observation for the current player.
  """
  current_player = observations['current_player']
  current_player_observation = (
      observations['player_observations'][current_player])

  legal_moves = current_player_observation['legal_moves_as_int']
  legal_moves = format_legal_moves(legal_moves, num_actions)

  observation_vector = current_player_observation['vectorized']
  obs_stacker.add_observation(observation_vector, current_player)
  observation_vector = obs_stacker.get_observation_stack(current_player)

  return current_player, legal_moves, observation_vector


def run_one_episode(agent, environment, obs_stacker):
  """Runs the agent on a single game of Hanabi in self-play mode.

  Args:
    agent: Agent playing Hanabi.
    environment: The Hanabi environment.
    obs_stacker: Observation stacker object.

  Returns:
    step_number: int, number of actions in this episode.
    total_reward: float, undiscounted return for this episode.
  """
  obs_stacker.reset_stack()
  observations = environment.reset()
  current_player, legal_moves, observation_vector = (
      parse_observations(observations, environment.num_moves(), obs_stacker))
  action = agent.begin_episode(current_player, legal_moves, observation_vector)

  is_done = False
  total_reward = 0
  step_number = 0

  has_played = {current_player}

  # Keep track of per-player reward.
  reward_since_last_action = np.zeros(environment.players)

  while not is_done:
    observations, reward, is_done, _ = environment.step(action.item())

    modified_reward = max(reward, 0) if LENIENT_SCORE else reward
    total_reward += modified_reward

    reward_since_last_action += modified_reward

    step_number += 1
    if is_done:
      break
    current_player, legal_moves, observation_vector = (
        parse_observations(observations, environment.num_moves(), obs_stacker))
    if current_player in has_played:
      action = agent.step(reward_since_last_action[current_player],
                          current_player, legal_moves, observation_vector)
    else:
      # Each player begins the episode on their first turn (which may not be
      # the first move of the game).
      action = agent.begin_episode(current_player, legal_moves,
                                   observation_vector)
      has_played.add(current_player)

    # Reset this player's reward accumulator.
    reward_since_last_action[current_player] = 0

  agent.end_episode(reward_since_last_action)

  tf.logging.info('EPISODE: %d %g', step_number, total_reward)
  return step_number, total_reward


def run_two_player_episode(agent_training, static_agent, environment, obs_stacker_online, obs_stacker_static):
    # Resets observation stacks and environment.
    obs_stacker_online.reset_stack()
    obs_stacker_static.reset_stack()
    observations = environment.reset()

    current_player, legal_moves, observation_vector = parse_observations(
        observations, environment.num_moves(),
        obs_stacker_online
    )

    is_done = False
    total_reward = 0
    step_number = 0
    reward_since_last_action = np.zeros(environment.players)

    while not is_done:
        # Select the correct agent and stacker based on the current player.
        if current_player == 0:
            # Training agent's turn
            action = agent_training.step(
                reward_since_last_action[current_player],
                current_player,
                legal_moves,
                observation_vector
            ) if step_number > 0 else agent_training.begin_episode(
                current_player, legal_moves, observation_vector
            )
        else:
            # Static agent's turn
            action = static_agent.step(
                reward_since_last_action[current_player],
                current_player,
                legal_moves,
                observation_vector
            ) if step_number > 0 else static_agent.begin_episode(
                current_player, legal_moves, observation_vector
            )

        # Environment step
        observations, reward, is_done, _ = environment.step(action.item())

        # Update rewards
        reward_since_last_action[current_player] += reward
        total_reward += reward
        step_number += 1

        # Prepare for the next turn if the game is not finished
        if not is_done:
            current_player, legal_moves, observation_vector = parse_observations(
                observations, environment.num_moves(),
                obs_stacker_online if current_player == 0 else obs_stacker_static
            )

    # End the episode for both agents
    agent_training.end_episode(reward_since_last_action)
    static_agent.end_episode(reward_since_last_action)

    return step_number, total_reward


def run_one_phase(online_agent, static_agent, environment, 
                  obs_stacker_online, obs_stacker_static, 
                  min_steps, statistics, run_mode_str):
    """Runs the two-agent game loop until a desired number of steps.

    Args:
        online_agent: The training agent.
        static_agent: The fixed opponent agent.
        environment: The Hanabi environment.
        obs_stacker_online: Observation stacker for the online agent.
        obs_stacker_static: Observation stacker for the static agent.
        min_steps: int, minimum number of steps to generate in this phase.
        statistics: `IterationStatistics` object to record results.
        run_mode_str: str, describes the run mode for this phase.

    Returns:
        step_count: Total steps taken in this phase.
        sum_returns: Total returns (rewards).
        num_episodes: Number of episodes completed.
    """
    step_count = 0
    num_episodes = 0
    sum_returns = 0.0

    while step_count < min_steps:
        episode_length, episode_return = run_two_player_episode(
            online_agent, static_agent, environment, obs_stacker_online, obs_stacker_static)
        
        statistics.append({
            f'{run_mode_str}_episode_lengths': episode_length,
            f'{run_mode_str}_episode_returns': episode_return
        })

        step_count += episode_length
        sum_returns += episode_return
        num_episodes += 1

    return step_count, sum_returns, num_episodes



@gin.configurable
def run_one_iteration(online_agent, static_agent, environment, 
                      obs_stacker_online, obs_stacker_static, iteration,
                      training_steps, evaluate_every_n=100, num_evaluation_games=100):
    """Runs one iteration of agent interaction, including training and evaluation.

    Args:
        online_agent: The training agent.
        static_agent: The fixed opponent agent.
        environment: The Hanabi environment.
        obs_stacker_online: Observation stacker for the online agent.
        obs_stacker_static: Observation stacker for the static agent.
        iteration: int, current iteration number.
        training_steps: int, number of training steps to perform.
        evaluate_every_n: int, evaluation frequency.
        num_evaluation_games: int, number of evaluation games.

    Returns:
        statistics: Summary statistics for the iteration.
    """
    start_time = time.time()
    statistics = iteration_statistics.IterationStatistics()

    # Training phase
    online_agent.eval_mode = False
    step_count, sum_returns, num_episodes = run_one_phase(
        online_agent, static_agent, environment, 
        obs_stacker_online, obs_stacker_static,
        training_steps, statistics, 'train')

    tf.logging.info(f'Training: Iteration {iteration} completed in {time.time() - start_time:.2f} seconds.')
    average_return = sum_returns / num_episodes
    statistics.append({'average_training_return': average_return})

    # Evaluation phase
    if evaluate_every_n is not None and iteration % evaluate_every_n == 0:
        tf.logging.info(f'Starting evaluation for iteration {iteration}.')
        online_agent.eval_mode = True
        total_rewards = []
        
        for _ in range(num_evaluation_games):
            _, total_reward = run_two_player_episode(
                online_agent, static_agent, environment, 
                obs_stacker_online, obs_stacker_static)
            total_rewards.append(total_reward)
        
        eval_average_return = np.mean(total_rewards)
        statistics.append({'eval_average_return': eval_average_return})
        tf.logging.info(f'Evaluation: Iteration {iteration} average return: {eval_average_return:.2f}')
    
    return statistics







def log_experiment(experiment_logger, iteration, statistics,
                   logging_file_prefix='log', log_every_n=1):
  """Records the results of the current iteration.

  Args:
    experiment_logger: A `Logger` object.
    iteration: int, iteration number.
    statistics: Object containing statistics to log.
    logging_file_prefix: str, prefix to use for the log files.
    log_every_n: int, specifies logging frequency.
  """
  if iteration % log_every_n == 0:
    experiment_logger['iter{:d}'.format(iteration)] = statistics
    experiment_logger.log_to_file(logging_file_prefix, iteration)


def checkpoint_experiment(experiment_checkpointer, agent, experiment_logger,
                          iteration, checkpoint_dir, checkpoint_every_n):
  """Checkpoint experiment data.

  Args:
    experiment_checkpointer: A `Checkpointer` object.
    agent: An RL agent.
    experiment_logger: a Logger object, to include its data in the checkpoint.
    iteration: int, iteration number for checkpointing.
    checkpoint_dir: str, the directory where to save checkpoints.
    checkpoint_every_n: int, the frequency for writing checkpoints.
  """
  if iteration % checkpoint_every_n == 0:
    agent_dictionary = agent.bundle_and_checkpoint(checkpoint_dir, iteration)
    if agent_dictionary:
      agent_dictionary['current_iteration'] = iteration
      agent_dictionary['logs'] = experiment_logger.data
      experiment_checkpointer.save_checkpoint(iteration, agent_dictionary)






@gin.configurable
def run_experiment(online_agent, static_agent, environment, 
                   start_iteration, obs_stacker_online, obs_stacker_static,
                   experiment_logger, experiment_checkpointer, checkpoint_dir,
                   num_iterations=200, training_steps=5000, 
                   logging_file_prefix='log', log_every_n=1, checkpoint_every_n=1,
                    benchmark_games=50):
    """Runs a full training experiment with two agents playing Hanabi.

    Args:
        online_agent: The training agent.
        static_agent: The fixed opponent agent.
        environment: The Hanabi environment.
        start_iteration: int, iteration to start from.
        obs_stacker_online: Observation stacker for the online agent.
        obs_stacker_static: Observation stacker for the static agent.
        experiment_logger: Logger for recording statistics.
        experiment_checkpointer: Checkpointer for saving progress.
        checkpoint_dir: Directory to save checkpoints.
        num_iterations: int, number of training iterations.
        training_steps: int, training steps per iteration.
        logging_file_prefix: str, prefix for log files.
        log_every_n: int, log frequency.
        checkpoint_every_n: int, checkpoint frequency.
        static_agent_update_interval: int, interval to update the static agent.
        benchmark_games: int, number of games for benchmarking.

    Returns:
        None
    """
    tf.logging.info('Starting experiment...')
    if num_iterations <= start_iteration:
        tf.logging.warning(f'num_iterations ({num_iterations}) <= start_iteration ({start_iteration}).')
        return

    for iteration in range(start_iteration, num_iterations):
        tf.logging.info(f'Iteration {iteration} starting...')
        statistics = run_one_iteration(
            online_agent, static_agent, environment, 
            obs_stacker_online, obs_stacker_static, 
            iteration, training_steps)

        # Logging and checkpointing
        log_experiment(experiment_logger, iteration, statistics,
                       logging_file_prefix, log_every_n)
        checkpoint_experiment(experiment_checkpointer, online_agent, experiment_logger,
                              iteration, checkpoint_dir, checkpoint_every_n)

    # Benchmarking trained agent
    # Benchmarking trained agent
    tf.logging.info('Benchmarking the trained agent...')
    benchmark_agent(online_agent, static_agent, environment, 
                    obs_stacker_online, obs_stacker_static, benchmark_games)



def benchmark_agent(online_agent, static_agent, environment, 
                    obs_stacker_online, obs_stacker_static, num_games=50):
    """Benchmarks two agents by letting them play against each other.

    Args:
        online_agent: The training agent.
        static_agent: The fixed opponent agent.
        environment: The Hanabi environment.
        obs_stacker_online: Observation stacker for the online agent.
        obs_stacker_static: Observation stacker for the static agent.
        num_games: int, number of benchmarking games.

    Returns:
        None
    """
    online_agent_total_reward = 0
    static_agent_total_reward = 0

    for game_idx in range(num_games):
        # Reset the stackers for the game
        obs_stacker_online.reset_stack()
        obs_stacker_static.reset_stack()

        # Run a single game with the two agents
        _, total_reward = run_two_player_episode(
            online_agent, static_agent, environment, 
            obs_stacker_online, obs_stacker_static)

        # The reward in Hanabi is shared, so both agents receive the same reward
        online_agent_total_reward += total_reward
        static_agent_total_reward += total_reward

        tf.logging.info(f'Benchmark Game {game_idx + 1}: Reward = {total_reward}')

    # Compute averages
    average_reward = online_agent_total_reward / num_games
    tf.logging.info(f'Benchmarking complete: Average reward over {num_games} games: {average_reward:.2f}')
    tf.logging.info(f'Total reward for Online Agent: {online_agent_total_reward}')
    tf.logging.info(f'Total reward for Static Agent: {static_agent_total_reward}')
