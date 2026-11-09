
import tkinter as tk

from src.agent.templates.control_algorithms.temporal_difference.q_learning import QLearning
from src.agent.common.policies import get_greedy_action
from src.env.gridworld import Gridworld
from src.env.gym import Gym


CELL_SIZE = 75
STEP_DELAY = 500  # Milliseconds between robot moves
TRAINING_EPISODES = 1000


class GridworldViewer:
    def __init__(self):
        self.gridworld = Gridworld()
        self.env = Gym(self.gridworld)
        self.agent = QLearning(self.env)

        # Train without launching the graph windows
        print("Training agent...")
        for _ in range(TRAINING_EPISODES):
            self.agent.run_iteration()
        print("Training complete!")

        self.root = tk.Tk()
        self.root.title("Q-Learning: Watch the Robot")

        self.canvas = tk.Canvas(
            self.root,
            width=self.env.cols * CELL_SIZE,
            height=self.env.rows * CELL_SIZE,
            bg="white"
        )
        self.canvas.pack(padx=10, pady=10)

        self.status = tk.Label(
            self.root,
            text="Click Watch Agent to see the learned path.",
            font=("Arial", 13)
        )
        self.status.pack(pady=5)

        self.watch_button = tk.Button(
            self.root,
            text="Watch Agent",
            command=self.watch_agent,
            font=("Arial", 13)
        )
        self.watch_button.pack(pady=10)

        self.current_state = self.env.reset()
        self.steps = 0
        self.running = False

        self.draw_grid()

    def draw_grid(self):
        self.canvas.delete("all")

        for (row, col), state in self.env.all_states.items():
            x1 = col * CELL_SIZE
            y1 = row * CELL_SIZE
            x2 = x1 + CELL_SIZE
            y2 = y1 + CELL_SIZE

            color = "white"

            if state.wall:
                color = "gray"
            elif state.terminal:
                color = "lightgreen" if state.reward >= 0 else "salmon"

            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                fill=color,
                outline="black"
            )

            if state.terminal:
                self.canvas.create_text(
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                    text=str(state.reward),
                    font=("Arial", 12)
                )

        # Find the robot's current coordinates
        for (row, col), state in self.env.all_states.items():
            if state == self.current_state:
                cx = col * CELL_SIZE + CELL_SIZE / 2
                cy = row * CELL_SIZE + CELL_SIZE / 2

                radius = CELL_SIZE / 4

                self.canvas.create_oval(
                    cx - radius,
                    cy - radius,
                    cx + radius,
                    cy + radius,
                    fill="dodgerblue",
                    outline="black",
                    width=2
                )
                self.canvas.create_text(
                    cx, cy,
                    text="R",
                    fill="white",
                    font=("Arial", 15, "bold")
                )
                break

    def watch_agent(self):
        if self.running:
            return

        self.current_state = self.env.reset()
        self.steps = 0
        self.running = True
        self.watch_button.config(state="disabled")

        self.draw_grid()
        self.root.after(STEP_DELAY, self.move_robot)

    def move_robot(self):
        if not self.running:
            return

        if self.current_state.terminal:
            self.finish("Goal reached!")
            return

        if self.steps >= self.env.episode_threshold:
            self.finish("Episode timed out.")
            return

        # Choose the best action learned by Q-Learning
        action = get_greedy_action(
            self.agent.action_values,
            self.current_state
        )

        # Move one step
        new_state, reward, terminated, truncated = self.env.step(action)

        self.current_state = new_state
        self.steps += 1

        self.draw_grid()
        self.status.config(
            text=f"Step {self.steps} | Action: {action}"
        )

        if terminated or truncated:
            self.finish(
                "Goal reached!" if terminated else "Episode timed out."
            )
        else:
            # Schedule the next movement without freezing Tkinter
            self.root.after(STEP_DELAY, self.move_robot)

    def finish(self, message):
        self.running = False
        self.status.config(text=message)
        self.watch_button.config(state="normal")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    viewer = GridworldViewer()
    viewer.run()
