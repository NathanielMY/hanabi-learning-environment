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
"""The entry point for running a Rainbow agent on Hanabi."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

from third_party.dopamine import logger

import run_experiment

FLAGS = flags.FLAGS

flags.DEFINE_multi_string(
    'gin_files', [],
    'List of paths to gin configuration files (e.g.'
    '"configs/hanabi_rainbow.gin").')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1").')

flags.DEFINE_string('base_dir', None,
                    'Base directory to host all required sub-directories.')

flags.DEFINE_string('checkpoint_dir', '',
                    'Directory where checkpoint files should be saved. If '
                    'empty, no checkpoints will be saved.')
flags.DEFINE_string('checkpoint_file_prefix', 'ckpt',
                    'Prefix to use for the checkpoint files.')
flags.DEFINE_string('logging_dir', '',
                    'Directory where experiment data will be saved. If empty '
                    'no checkpoints will be saved.')
flags.DEFINE_string('logging_file_prefix', 'log',
                    'Prefix to use for the log files.')


def launch_experiment():
  """Launches the experiment.

  Specifically:
  - Load the gin configs and bindings.
  - Initialize the Logger object.
  - Initialize the environment.
  - Initialize the observation stacker.
  - Initialize the agent.
  - Reload from the latest checkpoint, if available, and initialize the
    Checkpointer object.
  - Run the experiment.
  """
  if FLAGS.base_dir == None:
    raise ValueError('--base_dir is None: please provide a path for '
                     'logs and checkpoints.')

  run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
  experiment_logger = logger.Logger('{}/logs'.format(FLAGS.base_dir))

  print("got here")
  environment = run_experiment.create_environment()
  obs_stacker = run_experiment.create_obs_stacker(environment)
  agent = run_experiment.create_agent(environment, obs_stacker)

  # FLAG - # Warm-up phase to populate replay memory
  print("Starting warm-up phase to populate replay memory...")
  for _ in range(agent.min_replay_history):
      observation = environment.reset()
      done = False
      while not done:
          # Take random actions to populate replay memory
          legal_actions = environment.legal_moves()
          action = environment.random_action()  # Replace with proper random action logic if needed
          next_observation, reward, done, _ = environment.step(action)

          # Add the transition to replay memory
          agent._replay.memory._add(observation, action, reward, done, legal_actions)

          # Update observation
          observation = next_observation

  print(f"Warm-up phase completed. Replay memory populated with {agent.min_replay_history} transitions.")

  agent._replay.transition = agent._replay.get_transition() 


  checkpoint_dir = '{}/checkpoints'.format(FLAGS.base_dir)
  start_iteration, experiment_checkpointer = (
      run_experiment.initialize_checkpointing(agent,
                                              experiment_logger,
                                              checkpoint_dir,
                                              FLAGS.checkpoint_file_prefix))

  run_experiment.run_experiment(agent, environment, start_iteration,
                                obs_stacker,
                                experiment_logger, experiment_checkpointer,
                                checkpoint_dir,
                                logging_file_prefix=FLAGS.logging_file_prefix)


def main(unused_argv):
  """This main function acts as a wrapper around a gin-configurable experiment.

  Args:
    unused_argv: Arguments (unused).
  """
  launch_experiment()

if __name__ == '__main__':
  app.run(main)
