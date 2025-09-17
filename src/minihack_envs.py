import pathlib
from typing import Any, Literal, Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from minihack.envs.room import MiniHackRoom
from nle import nethack

ACTION = nethack.CompassCardinalDirection
ACTIONS = tuple(ACTION)
AGENT = 64
FREE = 46
GOAL = 62
START = 60

action_space = gym.spaces.Discrete(len(ACTIONS))

ObsType = dict[Literal["chars"] | Literal["chars", "pixel"], np.ndarray]
InfoType = dict[Literal["end_status", "is_ascended"], Any]
State = tuple[ObsType, InfoType]
HashableState = bytes
ActIndex = int


EMPTY_ROOM = "empty-room"
ROOM_WITH_LAVA = "room-with-lava"
ROOM_WITH_MONSTER = "room-with-monster"
ROOM_WITH_MULTIPLE_MONSTERS = "room-with-multiple-monsters"
CLIFF = "cliff-minihack"

des_room_with_lava = """
MAZE: "mylevel", ' '
FLAGS:premapped
GEOMETRY:center,center
MAP
|-----     ------
|.....-- --.....|
|.T.T...-.....L.|
|...T...........|
|.T.T...-.....L.|
|.....-----.....|
|-----     ------
ENDMAP
"""

des_cliff = """
MAZE: "mylevel", ' '
FLAGS:premapped
GEOMETRY:center,center
MAP
|----------------
|...............|
|...............|
|...............|
|...............|
|.LLLLLLLLLLLLL.|
|----------------
ENDMAP
"""

des_monster = """
MAZE: "mylevel", ' '
FLAGS:premapped
GEOMETRY:center,center
MAP
.....
.....
.....
.....
.....
ENDMAP
REGION: (0,0,20,80), lit, "ordinary"
MONSTER: ('Z', "ghoul"), (2,4)
"""


def get_env(
    id: str,
    *,
    add_pixels: bool = False,
    max_episode_steps: int = 1000,
    random: bool = False,
    size: int = 5,
) -> MiniHackRoom:
    observation_keys = ["chars", "pixel"] if add_pixels else ["chars"]

    if id == EMPTY_ROOM:
        env = MiniHackRoom(
            size=size,
            max_episode_steps=50,
            actions=ACTIONS,
            random=random,
            observation_keys=observation_keys,
            reward_win=0,
            penalty_step=-1,
            penalty_time=-1,
        )
    elif id == ROOM_WITH_LAVA:
        des_file = des_room_with_lava
        if not random:
            cont = """STAIR:(15,3),down \nBRANCH: (3,3,3,3),(4,4,4,4) \n"""
            des_file += cont

        env = gym.make(
            "MiniHack-Navigation-Custom-v0",
            actions=ACTIONS,
            des_file=des_file,
            max_episode_steps=max_episode_steps,
            observation_keys=observation_keys,
        )

        env = DoNotResetWhenDead(env, max_episode_steps)
    elif id == CLIFF:
        des_file = des_cliff
        if not random:
            cont = """STAIR:(15,5),down \nBRANCH: (1,5,1,5),(4,4,4,4) \n"""
            des_file += cont

        env = gym.make(
            "MiniHack-Navigation-Custom-v0",
            actions=ACTIONS,
            des_file=des_file,
            max_episode_steps=max_episode_steps,
            observation_keys=observation_keys,
        )
        env = DoNotResetWhenDead(env, max_episode_steps)
    elif id == ROOM_WITH_MONSTER:
        des_file = des_monster
        if not random:
            cont = """STAIR:(4,4),down \nBRANCH: (0,0,0,0),(1,1,1,1) \n"""
            des_file += cont

        env = gym.make(
            "MiniHack-Navigation-Custom-v0",
            actions=ACTIONS,
            des_file=des_file,
            max_episode_steps=max_episode_steps,
            observation_keys=observation_keys,
        )
        env = DoNotResetWhenDead(env, max_episode_steps)
    elif id == ROOM_WITH_MULTIPLE_MONSTERS:
        size = 7
        env = MiniHackRoom(
            size=size,
            max_episode_steps=100,
            actions=ACTIONS,
            random=random,
            n_monster=3,
            observation_keys=observation_keys,
            reward_win=0,
            reward_lose=-10,
            penalty_step=-1,
            penalty_time=-1,
        )
    else:
        raise Exception("Environment %s not found" % str(id))

    return env


class DoNotResetWhenDead(gym.Wrapper):
    def __init__(self, env, max_episode_steps=1000, goal_reward=0, negative_step_reward=-1, dead_negative_reward=-100):
        super().__init__(env)
        self.step_counter = 0
        self.max_episode_steps = max_episode_steps
        self.goal_reward = goal_reward
        self.negative_step_reward = negative_step_reward
        self.dead_negative_reward = dead_negative_reward

    def step(self, action, **kwargs):
        self.step_counter += 1
        observation, reward, terminated, truncated, info = self.env.step(action)

        if isinstance(self.env, MiniHackRoom):
            env = self.env.unwrapped
        else:
            env = self.env.env.unwrapped

        if info["end_status"] == env.StepStatus.ABORTED or self.step_counter == self.max_episode_steps:
            terminated = truncated = True
            reward = self.negative_step_reward
        elif info["end_status"] == env.StepStatus.DEATH:
            terminated = truncated = False
            reward = self.dead_negative_reward + self.negative_step_reward
            observation, info = self.env.reset(**kwargs)
        elif info["end_status"] == env.StepStatus.TASK_SUCCESSFUL:
            terminated = truncated = True
            reward = self.goal_reward
        else:
            terminated = truncated = False
            reward = self.negative_step_reward

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.step_counter = 0
        return observation, info


BLANK = 32
PIXEL_SIZE = 16


def get_crop_chars_from_observation(observation: ObsType) -> np.ndarray:
    chars = observation["chars"]
    coords = np.argwhere(chars != BLANK)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    chars = chars[x_min : x_max + 1, y_min : y_max + 1]
    return chars


def get_crop_pixel_from_observation(observation: ObsType) -> np.ndarray:
    coords = np.argwhere(observation["chars"] != BLANK)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    non_empty_pixels = observation["pixel"][
        x_min * PIXEL_SIZE : (x_max + 1) * PIXEL_SIZE,
        y_min * PIXEL_SIZE : (y_max + 1) * PIXEL_SIZE,
    ]
    return non_empty_pixels


def plot_observations(state: State, save_fig: Optional[pathlib.Path] = None) -> None:
    observation, _ = state
    chars = get_crop_chars_from_observation(observation)
    pixels = get_crop_pixel_from_observation(observation)

    print(chars)
    plt.imshow(pixels)
    if save_fig:
        save_fig_path = pathlib.Path(__file__).resolve().parent.parent / "tex" / "figures" / save_fig
        save_fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{save_fig_path}.png")
    plt.show()


def hashable(state: State) -> HashableState:
    observation, _ = state
    chars = get_crop_chars_from_observation(observation)

    return chars.data.tobytes()
