from abc import ABC, abstractmethod
import numpy as np
from typing import Union, Dict, Any
from gymnasium import Env

# Type aliases for clarity
State = Union[np.ndarray, int, float]  # Can be array (Box) or scalar (Discrete)
Action = Union[int, np.ndarray]  # Can be int (Discrete) or array (Box)
Reward = float
Done = bool
Info = Dict[str, Any]

class OnlineAgent(ABC):
    @abstractmethod
    def step(self, state: State) -> Action:
        raise NotImplementedError
    @abstractmethod
    def update(self, state: State, next_state: State, reward: float, done: bool):
        raise NotImplementedError

class OfflineAgent(ABC):
    @abstractmethod
    def offline_update(self, env: Env):
        raise NotImplementedError