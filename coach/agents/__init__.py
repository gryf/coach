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
from coach.agents.actor_critic_agent import ActorCriticAgent
from coach.agents.agent import Agent
from coach.agents.bc_agent import BCAgent
from coach.agents.bootstrapped_dqn_agent import BootstrappedDQNAgent
from coach.agents.categorical_dqn_agent import CategoricalDQNAgent
from coach.agents.clipped_ppo_agent import ClippedPPOAgent
from coach.agents.ddpg_agent import DDPGAgent
from coach.agents.ddqn_agent import DDQNAgent
from coach.agents.dfp_agent import DFPAgent
from coach.agents.dqn_agent import DQNAgent
from coach.agents.human_agent import HumanAgent
from coach.agents.imitation_agent import ImitationAgent
from coach.agents.mmc_agent import MixedMonteCarloAgent
from coach.agents.n_step_q_agent import NStepQAgent
from coach.agents.naf_agent import NAFAgent
from coach.agents.nec_agent import NECAgent
from coach.agents.pal_agent import PALAgent
from coach.agents.policy_gradients_agent import PolicyGradientsAgent
from coach.agents.policy_optimization_agent import PolicyOptimizationAgent
from coach.agents.ppo_agent import PPOAgent
from coach.agents.qr_dqn_agent import QuantileRegressionDQNAgent
from coach.agents.value_optimization_agent import ValueOptimizationAgent

__all__ = [ActorCriticAgent,
           Agent,
           BCAgent,
           BootstrappedDQNAgent,
           CategoricalDQNAgent,
           ClippedPPOAgent,
           DDPGAgent,
           DDQNAgent,
           DFPAgent,
           DQNAgent,
           HumanAgent,
           ImitationAgent,
           MixedMonteCarloAgent,
           NAFAgent,
           NECAgent,
           NStepQAgent,
           PALAgent,
           PPOAgent,
           PolicyGradientsAgent,
           PolicyOptimizationAgent,
           QuantileRegressionDQNAgent,
           ValueOptimizationAgent]
