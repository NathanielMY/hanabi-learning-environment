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

import run_experiment_same_agent

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


# def launch_experiment():
#   """Launches the experiment.

#   Specifically:
#   - Load the gin configs and bindings.
#   - Initialize the Logger object.
#   - Initialize the environment.
#   - Initialize the observation stacker.
#   - Initialize the agent.
#   - Reload from the latest checkpoint, if available, and initialize the
#     Checkpointer object.
#   - Run the experiment.
#   """
#   if FLAGS.base_dir == None:
#     raise ValueError('--base_dir is None: please provide a path for '
#                      'logs and checkpoints.')

#   run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
#   experiment_logger = logger.Logger('{}/logs'.format(FLAGS.base_dir))

#   environment = run_experiment.create_environment()
#   obs_stacker = run_experiment.create_obs_stacker(environment)
  

#   checkpoint_dir = '{}/checkpoints'.format(FLAGS.base_dir)
#   online_agent = run_experiment.create_agent(environment, obs_stacker)
#   start_iteration, experiment_checkpointer = (
#     run_experiment.initialize_checkpointing(
#         online_agent,
#         experiment_logger,
#         checkpoint_dir,
#         FLAGS.checkpoint_file_prefix
#       )
#     )
  
#   static_agent = run_experiment.create_agent(environment, obs_stacker)

#   run_experiment.run_experiment(online_agent, static_agent, environment, start_iteration,
#                                 obs_stacker,
#                                 experiment_logger, experiment_checkpointer,
#                                 checkpoint_dir,
#                                 logging_file_prefix=FLAGS.logging_file_prefix)
  


def launch_experiment_selfplay():
    """Launches the self-play experiment.

    Specifically:
    - Load the gin configs and bindings.
    - Initialize the Logger object.
    - Initialize the environment.
    - Initialize the observation stackers.
    - Initialize the online (training) and static (fixed) agents.
    - Reload from the latest checkpoint, if available, and initialize the
      Checkpointer object.
    - Run the experiment with training and benchmarking.
    """
    if FLAGS.base_dir is None:
        raise ValueError('--base_dir is None: please provide a path for '
                         'logs and checkpoints.')

    # Load configurations
    run_experiment_same_agent.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)

    # Initialize logger
    experiment_logger = logger.Logger(f'{FLAGS.base_dir}/logs')

    # Create environment
    environment = run_experiment_same_agent.create_environment()

    # Create observation stackers
    obs_stacker_online = run_experiment_same_agent.create_obs_stacker(environment)
    obs_stacker_static = run_experiment_same_agent.create_obs_stacker(environment)

    # Set up checkpoint directory
    checkpoint_dir = f'{FLAGS.base_dir}/checkpoints'

    # Initialize online (training) agent
    online_agent = run_experiment_same_agent.create_agent(environment, obs_stacker_online)

    # Reload from the latest checkpoint (if available) and initialize checkpointer
    start_iteration, experiment_checkpointer = (
        run_experiment_same_agent.initialize_checkpointing(
            online_agent,
            experiment_logger,
            checkpoint_dir,
            FLAGS.checkpoint_file_prefix
        )
    )

    # Initialize static (fixed opponent) agent
    static_agent = run_experiment_same_agent.create_agent(environment, obs_stacker_static)

    # Run the experiment
    run_experiment_same_agent.run_experiment(
        online_agent=online_agent,
        static_agent=static_agent,
        environment=environment,
        start_iteration=start_iteration,
        obs_stacker_online=obs_stacker_online,
        obs_stacker_static=obs_stacker_static,
        experiment_logger=experiment_logger,
        experiment_checkpointer=experiment_checkpointer,
        checkpoint_dir=checkpoint_dir,
        logging_file_prefix=FLAGS.logging_file_prefix
    )


def main(unused_argv):
  """This main function acts as a wrapper around a gin-configurable experiment.

  Args:
    unused_argv: Arguments (unused).
  """
  launch_experiment_selfplay()

if __name__ == '__main__':
  app.run(main)
