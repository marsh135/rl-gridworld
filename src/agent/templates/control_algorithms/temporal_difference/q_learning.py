from pathlib import Path

import numpy as np
import yaml

from src.agent.common.policies import Policy, create_epsilon_soft_policy
from src.agent.common.value_functions import ActionValueFunction
from src.env.action import Action
from src.env.gym import Gym
from src.env.state import State
from src.visualization.plotter import plot_returns, plot_value_function_sums


class QLearning:
    """
        This is the Temporal Difference learning algorithm Sarsa
        There is no domain knowledge available
        The action values should be updated during episode run (online)

        For reference, see Sutton & Barto, Reinforcement Learning: An Introduction, 2018, p. 131, chapter 6.5, Q-Learning: Off-policy TD Control.
    """

    def __init__(self, env):
        self.algo_name: str = "Q-Learning"
        with open(Path(__file__).parent.parent / 'algorithms-config.yml') as f:
            self.config = yaml.safe_load(f)[self.algo_name]
        self.env: Gym = env
        self.discount_factor: float = self.config['discount_factor']
        self.step_size: float = self.config['step_size']
        self.epsilon: float = self.config['epsilon']
        self.epsilon_decay: float = self.config['epsilon_decay']
        self._total_episodes = 0
        self._get_iteration_size: int = self.config['iterations']
        self._rng = np.random.default_rng(seed=1)
        self._plot_returns: bool = self.config['plot_returns']
        self._plot_value_functions: bool = self.config['plot_value_functions']
        self._returns: list[float] = []
        self._value_functions_sum: list[float] = []
        self.action_values: ActionValueFunction = ActionValueFunction(env=self.env, init_value=self.config['value_function_init'])
        self.policy: Policy = create_epsilon_soft_policy(self.action_values, self.epsilon)

    def clear(self) -> None:
        self.env.clear()
        self._returns = []
        self._total_episodes = 0
        self._value_functions_sum = []
        self.epsilon = self.config['epsilon']
        self.action_values = ActionValueFunction(env=self.env, init_value=self.config['value_function_init'])
        self.policy = create_epsilon_soft_policy(self.action_values, self.epsilon)

    def run(self) -> None:
        for _ in range(self._get_iteration_size):
            self.run_iteration()
        if self._plot_returns or self._plot_value_functions:
            self.render()

    def render(self) -> None:
        if self._plot_value_functions:
            plot_value_function_sums(self.algo_name, self._value_functions_sum)
        if self._plot_returns:
            plot_returns(self.algo_name, self._returns)

    def run_iteration(self) -> None:
        self.run_episode()
        self.policy = create_epsilon_soft_policy(self.action_values, self.epsilon)
        self._value_functions_sum.append(self.action_values.sum())

    def run_episode(self) -> None:
        # Reset the environment to the starting state
        state = self.env.reset()

        # Track the total reward earned during this episode
        total_reward = 0.0

        # Continue until the goal is reached or time runs out
        done = False

        while not done:

            # Update the epsilon-greedy policy using current Q-values
            self.policy = create_epsilon_soft_policy(
                self.action_values, self.epsilon
            )

            # Select an action (exploration or exploitation)
            action = self.policy.select_action(state)

            # Execute the action in the environment
            new_state, reward, terminated, truncated = self.env.step(action)

            # Update the Q-table using the observed reward
            self.update_action_value(
                state, action, reward, new_state
            )

            # Accumulate rewards
            total_reward += reward

            # Reduce exploration over time
            self.epsilon = max(
                0.0, min(1.0, self.epsilon * self.epsilon_decay)
            )

            # Move to the next state
            state = new_state

            # Stop when the goal is reached or the episode times out
            done = terminated or truncated

        # Record episode results for plotting
        print(f"Episode {self._total_episodes}: Reward = {total_reward:.2f}")
        self._returns.append(total_reward)
        self._total_episodes += 1

    def update_action_value(self, state, action, reward, new_state) -> None:
        new_value = self.calculate_action_value(
            state, action, reward, new_state
        )

        self.action_values.update_value(
            state, action, new_value
        )


    def calculate_action_value(
        self, s: State, a: Action, r: float, new_s: State
    ) -> float:
        old_q = self.action_values.get(s, a)
        error = self.q_learning_error(s, a, r, new_s)

        return old_q + self.step_size * error


    def q_learning_error(
        self, s: State, a: Action, r: float, new_s: State
    ) -> float:
        target = self.q_learning_target(r, new_s)
        old_q = self.action_values.get(s, a)

        return target - old_q


    def q_learning_target(self, reward: float, new_s: State) -> float:
        if new_s.terminal:
            return reward

        best_future_q = max(
            self.action_values.get(new_s, action)
            for action in self.env.actions
        )

        return reward + self.discount_factor * best_future_q
