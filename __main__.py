from src.agent.templates.control_algorithms.temporal_difference.q_learning import QLearning
from src.env.gridworld import Gridworld
from src.env.gym import Gym
from src.visualization.rl_board import RlBoard


def main():
    gridworld = Gridworld()
    env = Gym(gridworld)

    agents = [QLearning(env)]

    board = RlBoard(agents, env)
    board.show()


if __name__ == "__main__":
    main()