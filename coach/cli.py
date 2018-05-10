#
# Copyright (c) 2017 Intel Corporation
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
import argparse
import atexit
import json
import os
import re
import subprocess
import sys
import time

import tensorflow as tf

from coach import agents  # noqa
from coach import configurations as conf
from coach import environments
from coach import logger
from coach import presets
from coach import utils


def check_input_and_fill_run_dict(parser):
    args = parser.parse_args()

    # if no arg is given
    if len(sys.argv) == 1:
        parser.print_help()
        exit(0)

    # list available presets
    if args.list:
        presets_lists = utils.list_all_classes_in_module(presets)
        logger.screen.log_title("Available Presets:")
        for preset in presets_lists:
            print(preset)
        sys.exit(0)

    preset_names = utils.list_all_classes_in_module(presets)
    if args.preset is not None and args.preset not in preset_names:
        logger.screen.error("A non-existing preset was selected. ")

    if (args.checkpoint_restore_dir is not None and not
            os.path.exists(args.checkpoint_restore_dir)):
        logger.screen.error("The requested checkpoint folder to load from "
                            "does not exist. ")

    if (not args.preset and not
        all([args.agent_type, args.environment_type,
             args.exploration_policy_type])):
        logger.screen.error('When no preset is given for Coach to run, the '
                            'user is expected to input the desired agent_type,'
                            ' environment_type and exploration_policy_type to'
                            ' assemble a preset.\nAt least one of these '
                            'parameters was not given.')

    # get experiment name and path
    experiment_name = logger.logger.get_experiment_name(args.experiment_name)
    experiment_path = logger.logger.get_experiment_path(experiment_name)

    # fill run_dict
    run_dict = dict()
    run_dict['agent_type'] = args.agent_type
    run_dict['environment_type'] = args.environment_type
    run_dict['exploration_policy_type'] = args.exploration_policy_type
    run_dict['level'] = args.level
    run_dict['preset'] = args.preset
    run_dict['custom_parameter'] = args.custom_parameter
    run_dict['experiment_path'] = experiment_path
    run_dict['evaluate'] = args.evaluate

    # multi-threading parameters
    run_dict['num_threads'] = args.num_workers

    # checkpoints
    run_dict['save_model_sec'] = args.save_model_sec
    run_dict['save_model_dir'] = None
    if args.save_model_sec:
        run_dict['save_model_dir'] = experiment_path
    run_dict['checkpoint_restore_dir'] = args.checkpoint_restore_dir

    # visualization
    run_dict['visualization.dump_gifs'] = args.dump_gifs
    run_dict['visualization.render'] = args.render
    run_dict['visualization.tensorboard'] = args.tensorboard

    return args, run_dict


def run_dict_to_json(_run_dict, task_id=''):
    if task_id != '':
        json_path = os.path.join(_run_dict['experiment_path'],
                                 'run_dict_worker{}.json'.format(task_id))
    else:
        json_path = os.path.join(_run_dict['experiment_path'], 'run_dict.json')

    with open(json_path, 'w') as outfile:
        json.dump(_run_dict, outfile, indent=2)

    return json_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--preset', default=None,
                        help='(string) Name of a preset to run (as configured '
                        'in presets.py)')
    parser.add_argument('-l', '--list', action='store_true',
                        help='(flag) List all available presets')
    parser.add_argument('-e', '--experiment_name', default='',
                        help='(string) Experiment name to be used to store '
                        'the results.')
    parser.add_argument('-r', '--render', action='store_true',
                        help='(flag) Render environment')
    parser.add_argument('-n', '--num_workers', default=1, type=int,
                        help='(int) Number of workers for multi-process based '
                        'agents, e.g. A3C')
    parser.add_argument('--evaluate', action='store_true',
                        help='(flag) Run evaluation only. This is a '
                        'convenient way to disable training in order to '
                        'evaluate an existing checkpoint.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='(flag) Don\'t suppress TensorFlow debug prints.')
    parser.add_argument('-s', '--save_model_sec', default=None, type=int,
                        help='(int) Time in seconds between saving checkpoints'
                        ' of the model.')
    parser.add_argument('-crd', '--checkpoint_restore_dir',
                        help='(string) Path to a folder containing a '
                        'checkpoint to restore the model from.')
    parser.add_argument('-dg', '--dump_gifs', action='store_true',
                        help='(flag) Enable the gif saving functionality.')
    parser.add_argument('-at', '--agent_type', default=None,
                        help='(string) Choose an agent type class to override'
                        ' on top of the selected preset. If no preset is '
                        'defined, a preset can be set from the command-line '
                        'by combining settings which are set by using '
                        '--agent_type, --experiment_type, --environemnt_type')
    parser.add_argument('-et', '--environment_type', default=None,
                        help='(string) Choose an environment type class to '
                        'override on top of the selected preset. If no preset'
                        ' is defined, a preset can be set from the '
                        'command-line by combining settings which are set by '
                        'using --agent_type, --experiment_type, '
                        '--environemnt_type')
    parser.add_argument('-ept', '--exploration_policy_type', default=None,
                        help='(string) Choose an exploration policy type '
                        'class to override on top of the selected preset. If '
                        'no preset is defined, a preset can be set from the '
                        'command-line by combining settings which are set by '
                        'using --agent_type, --experiment_type, '
                        '--environemnt_type')
    parser.add_argument('-lvl', '--level', default=None,
                        help='(string) Choose the level that will be played '
                        'in the environment that was selected. This value '
                        'will override the level parameter in the environment '
                        'class.')
    parser.add_argument('-cp', '--custom_parameter', default=None,
                        help='(string) Semicolon separated parameters used to '
                        'override specific parameters on top of the selected '
                        'preset (or on top of the command-line assembled '
                        'one). Whenever a parameter value is a string, it '
                        'should be inputted as "string". For ex.: '
                        '"visualization.render=False; '
                        'num_training_iterations=500; optimizer=\'rmsprop\'"')
    parser.add_argument('-pf', '--parameters_file', default=None,
                        help='YAML file with customized parameters, just like '
                        '\'--custom-parameter\' bit in a file for convenience')
    parser.add_argument('--print_parameters', action='store_true',
                        help='(flag) Print tuning_parameters to stdout')
    parser.add_argument('-tb', '--tensorboard', action='store_true',
                        help='(flag) When using the TensorFlow backend, '
                        'enable TensorBoard log dumps. ')
    parser.add_argument('-ns', '--no_summary', action='store_true',
                        help='(flag) Prevent Coach from printing a summary '
                        'and asking questions at the end of runs')

    args, run_dict = check_input_and_fill_run_dict(parser)

    # turn TF debug prints off
    if not args.verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # dump documentation
    logger.logger.set_dump_dir(run_dict['experiment_path'], add_timestamp=True)
    if not args.no_summary:
        atexit.register(logger.logger.summarize_experiment)
        logger.screen.change_terminal_title(logger.logger.experiment_name)

    # Single-threaded runs
    if run_dict['num_threads'] == 1:
        # set tuning parameters
        json_run_dict_path = run_dict_to_json(run_dict)
        tuning_parameters = presets.json_to_preset(json_run_dict_path)
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.2
        tuning_parameters.sess = tf.Session(config=config)

        if args.print_parameters:
            print('tuning_parameters', tuning_parameters)

        # Single-thread runs
        tuning_parameters.task_index = 0
        env_instance = environments.create_environment(tuning_parameters)  # noqa
        agent = eval('agents.' + tuning_parameters.agent.type +
                     '(env_instance, tuning_parameters)')

        # Start the training or evaluation
        if tuning_parameters.evaluate:
            # evaluate forever
            agent.evaluate(sys.maxsize, keep_networks_synced=True)
        else:
            agent.improve()

    # Multi-threaded runs
    else:
        os.environ['OMP_NUM_THREADS'] = '1'
        # set parameter server and workers addresses
        ps_hosts = 'localhost:{}'.format(utils.get_open_port())
        worker_hosts = ','.join(['localhost:{}'.format(utils.get_open_port())
                                 for i in range(run_dict['num_threads'] + 1)])

        # Make sure to disable GPU so that all the workers will use the CPU
        utils.set_cpu()

        # create a parameter server
        cmd = ["python3",
               "./parallel_actor.py",
               "--ps_hosts={}".format(ps_hosts),
               "--worker_hosts={}".format(worker_hosts),
               "--job_name=ps"]
        subprocess.Popen(cmd)

        logger.screen.log_title("*** Distributed Training ***")
        time.sleep(1)

        # create N training workers and 1 evaluating worker
        workers = []

        for i in range(run_dict['num_threads'] + 1):
            # this is the evaluation worker
            run_dict['task_id'] = i
            if i == run_dict['num_threads']:
                run_dict['evaluate_only'] = True
                run_dict['visualization.render'] = args.render
            else:
                run_dict['evaluate_only'] = False
                # In a parallel setting, only the evaluation agent renders
                run_dict['visualization.render'] = False

            json_run_dict_path = run_dict_to_json(run_dict, i)
            workers_args = ["python3", "./parallel_actor.py",
                            "--ps_hosts={}".format(ps_hosts),
                            "--worker_hosts={}".format(worker_hosts),
                            "--job_name=worker",
                            "--load_json={}".format(json_run_dict_path)]

            p = subprocess.Popen(workers_args)

            if i != run_dict['num_threads']:
                workers.append(p)
            else:
                evaluation_worker = p

        # wait for all workers
        [w.wait() for w in workers]
        evaluation_worker.kill()


if __name__ == "__main__":
    main()
