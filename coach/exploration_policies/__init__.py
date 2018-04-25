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
from coach.exploration_policies.additive_noise import AdditiveNoise
from coach.exploration_policies.approximated_thompson_sampling_using_dropout import ApproximatedThompsonSamplingUsingDropout
from coach.exploration_policies.bayesian import Bayesian
from coach.exploration_policies.boltzmann import Boltzmann
from coach.exploration_policies.bootstrapped import Bootstrapped
from coach.exploration_policies.categorical import Categorical
from coach.exploration_policies.continuous_entropy import ContinuousEntropy
from coach.exploration_policies.e_greedy import EGreedy
from coach.exploration_policies.exploration_policy import ExplorationPolicy
from coach.exploration_policies.greedy import Greedy
from coach.exploration_policies.ou_process import OUProcess
from coach.exploration_policies.thompson_sampling import ThompsonSampling


__all__ = [AdditiveNoise,
           ApproximatedThompsonSamplingUsingDropout,
           Bayesian,
           Boltzmann,
           Bootstrapped,
           Categorical,
           ContinuousEntropy,
           EGreedy,
           ExplorationPolicy,
           Greedy,
           OUProcess,
           ThompsonSampling]
