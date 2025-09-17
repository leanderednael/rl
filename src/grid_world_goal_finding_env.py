from typing import Any, Final, Optional

import gymnasium as gym
import numpy as np

from src.agents import ActIndex, Reward, State
from src.minihack_envs import (
    ACTION,
    ACTIONS,
    AGENT,
    FREE,
    GOAL,
    START,
    InfoType,
    ObsType,
)

action_space = gym.spaces.Discrete(len(ACTIONS))

Coordinates = tuple[int, int]


class Env(gym.Env[ObsType, ActIndex]):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, n: int, m: int, *, render_mode: str = "human") -> None:
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render mode: {render_mode}. See metadata 'render_modes'.")
        self.render_mode = render_mode

        self.num_rows: Final = n
        self.num_cols: Final = m

        self.start_position: Final[Coordinates] = (0, 0)
        self.goal_position: Final[Coordinates] = (n - 1, m - 1)
        self.reward_per_action: Final[float] = -1.0

        observation, _ = self.reset()
        rows, cols = np.where(observation["chars"] == AGENT)
        self.agent_position: Coordinates = (int(rows[0]), int(cols[0]))

    def step(self, action: ActIndex) -> tuple[ObsType, Reward, bool, bool, InfoType]:  # type: ignore[override]
        row, col = self.agent_position

        match ACTIONS[action]:
            case ACTION.N:
                row -= 1
            case ACTION.S:
                row += 1
            case ACTION.E:
                col += 1
            case ACTION.W:
                col -= 1

        if row >= 0 and col >= 0 and row < self.num_rows and col < self.num_cols:
            self.agent_position = (row, col)

        return (
            {"chars": self._get_chars()},
            self.reward_per_action,
            self.agent_position == self.goal_position,
            False,
            {},
        )

    def reset(  # type: ignore[override]
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> State:
        super().reset(seed=seed, options=options)

        self.agent_position = self.start_position

        return {"chars": self._get_chars()}, {}

    def render(self):
        grid = self._get_chars()

        if self.render_mode == "human":
            print(
                "\n".join(
                    [
                        grid.tobytes().decode("utf-8")[i : i + self.num_cols]
                        for i in range(0, self.num_rows * self.num_cols, self.num_cols)
                    ]
                )
                + "\n"
            )
            return None
        else:
            return grid

    def _get_chars(self) -> np.ndarray:
        grid = FREE * np.ones((self.num_rows, self.num_cols), dtype=np.int8)
        grid[self.start_position] = START
        grid[self.goal_position] = GOAL
        grid[self.agent_position] = AGENT
        return grid
