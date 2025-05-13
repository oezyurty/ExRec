from tianshou.env.venvs import BaseVectorEnv
from collections.abc import Callable, Sequence
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import torch

from tianshou.env.utils import ENV_TYPE, gym_new_venv_step_type
from tianshou.env.worker import (
    DummyEnvWorker,
    EnvWorker,
    RayEnvWorker,
    SubprocEnvWorker,
)

GYM_RESERVED_KEYS = [
    "metadata",
    "reward_range",
    "spec",
    "action_space",
    "observation_space",
]

from exercise_recommender.envs.vector_env_worker import VectorEnvWorker
from exercise_recommender.envs.cluster_vector_env import ClusterVectorEnv

class VectorEnvWrapper(BaseVectorEnv):

    def __init__(
        self,
        env: VectorEnvWorker | ClusterVectorEnv
    ) -> None:
        env_fns = [lambda: env]
        self.env = env
        self.env_batch = env.batch_size
        super().__init__(env_fns=env_fns, worker_fn=DummyEnvWorker, wait_num=None, timeout=None)
        self.env_num = self.env_batch
        assert len(self.workers) == 1, f"Number of workers should be 1"
        self.worker = self.workers[0]
        assert self.worker is not None, f"EnvWorker should not be None"
    
    def _wrap_id(
        self,
        id: int | list[int] | np.ndarray | None = None,
    ) -> list[int] | np.ndarray:
        """
            Can only have 1 ID and that is 0
        """
        if id is None:
            return list(range(self.env_num))
        return [0] if np.isscalar(id) else [0 for i in range(len(id))]

    def _assert_id(self, id: list[int] | np.ndarray) -> None:
        for i in id:
            assert i == 0, f"Environment ID's should be converted to 0"
        for i in id:
            assert (
                i not in self.waiting_id
            ), f"Cannot interact with environment {i} which is stepping now."
            assert i in self.ready_id, f"Can only interact with ready environments {self.ready_id}."
    
    def _assert_is_not_closed(self) -> None:
        assert (
            not self.is_closed
        ), f"Methods of {self.__class__.__name__} cannot be called after close."
    
    def __getattribute__(self, key: str) -> Any:
        """Switch the attribute getter depending on the key.

        Any class who inherits ``gym.Env`` will inherit some attributes, like
        ``action_space``. However, we would like the attribute lookup to go straight
        into the worker (in fact, this vector env's action_space is always None).
        """
        if key in GYM_RESERVED_KEYS:  # reserved keys in gym.Env
            return self.get_env_attr(key)
        return super().__getattribute__(key)
    
    def get_env_attr(
        self,
        key: str,
        id: int | list[int] | np.ndarray | None = None,
    ) -> list[Any]:
        """Get an attribute from the underlying environments.

        If id is an int, retrieve the attribute denoted by key from the environment
        underlying the worker at index id. The result is returned as a list with one
        element. Otherwise, retrieve the attribute for all workers at indices id and
        return a list that is ordered correspondingly to id.

        :param str key: The key of the desired attribute.
        :param id: Indice(s) of the desired worker(s). Default to None for all env_id.

        :return list: The list of environment attributes.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)
        
        return [self.worker.get_env_attr(key) for _ in id]
    
    def reset(
        self,
        env_id: int | list[int] | np.ndarray | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reset the state of some envs and return initial observations.

        If id is None, reset the state of all the environments and return
        initial observations, otherwise reset the specific environments with
        the given id, either an int or a list.
        """
        self._assert_is_not_closed()
        if env_id is None:
            env_id = list(range(self.env_num))
        if not isinstance(env_id, list) and not isinstance(env_id, np.ndarray):
            raise ValueError("Individual environment reset is not supported")
        if len(env_id) != self.env_batch:
            raise ValueError("Partial environment reset is not supported")
        if self.is_async:
            self._assert_id(env_id)
        # send(None) == reset() in worker
        self.worker.send(None, **kwargs)
        return_list = self.worker.recv()
        assert (
            isinstance(return_list, tuple | list)
            and len(return_list) == 2
            and isinstance(return_list[1], dict)
        ), "The environment does not adhere to the Gymnasium's API."

        obs_list = return_list[0] # will have shape 1024 by 1068

        if isinstance(obs_list[0], tuple):  # type: ignore
            raise TypeError(
                "Tuple observation space is not supported. ",
                "Please change it to array or dict space",
            )
        try:
            obs = np.stack(obs_list)
        except ValueError:  # different len(obs)
            obs = np.array(obs_list, dtype=object)
        if not return_list[1]:
            infos = np.array([return_list[1] for _ in range(self.env_num)])
        else:
            infos = return_list[1]
        return obs, infos
    
    def step(
        self,
        action: np.ndarray | torch.Tensor | None,
        id: int | list[int] | np.ndarray | None = None,
    ) -> gym_new_venv_step_type:
        """Run one timestep of some environments' dynamics.

        If id is None, run one timestep of all the environments` dynamics;
        otherwise run one timestep for some environments with given id,  either
        an int or a list. When the end of episode is reached, you are
        responsible for calling reset(id) to reset this environment`s state.

        Accept a batch of action and return a tuple (batch_obs, batch_rew,
        batch_done, batch_info) in numpy format.

        :param numpy.ndarray action: a batch of action provided by the agent.
            If the venv is async, the action can be None, which will result
            in all arrays in the returned tuple being empty.

        :return: A tuple consisting of either:

            * ``obs`` a numpy.ndarray, the agent's observation of current environments
            * ``rew`` a numpy.ndarray, the amount of rewards returned after \
                previous actions
            * ``terminated`` a numpy.ndarray, whether these episodes have been \
                terminated
            * ``truncated`` a numpy.ndarray, whether these episodes have been truncated
            * ``info`` a numpy.ndarray, contains auxiliary diagnostic \
                information (helpful for debugging, and sometimes learning)

        For the async simulation:

        Provide the given action to the environments. The action sequence
        should correspond to the ``id`` argument, and the ``id`` argument
        should be a subset of the ``env_id`` in the last returned ``info``
        (initially they are env_ids of all the environments). If action is
        None, fetch unfinished step() calls instead.
        """
        self._assert_is_not_closed()
        if not isinstance(id, list) and not isinstance(id, np.ndarray):
            print(type(id))
            raise ValueError("Individual environment step is not supported")
        if len(id) != self.env_batch:
            print(len(id))
            raise ValueError("Partial environment step is not supported")

        if not self.is_async:
            if action is None:
                raise ValueError("action must be not-None for non-async")
            assert len(action) == len(id)
            self.worker.send(action)
            result = self.worker.recv()
        else:
            raise ValueError("Async Environment is not supported")
            if action is not None:
                self._assert_id(id)
                assert len(action) == len(id)
                for act, env_id in zip(action, id, strict=True):
                    self.workers[env_id].send(act)
                    self.waiting_conn.append(self.workers[env_id])
                    self.waiting_id.append(env_id)
                self.ready_id = [x for x in self.ready_id if x not in id]
            ready_conns: list[EnvWorker] = []
            while not ready_conns:
                ready_conns = self.worker_class.wait(self.waiting_conn, self.wait_num, self.timeout)
            result = []
            for conn in ready_conns:
                waiting_index = self.waiting_conn.index(conn)
                self.waiting_conn.pop(waiting_index)
                env_id = self.waiting_id.pop(waiting_index)
                # env_return can be (obs, reward, done, info) or
                # (obs, reward, terminated, truncated, info)
                env_return = conn.recv()
                env_return[-1]["env_id"] = env_id  # Add `env_id` to info
                result.append(env_return)
                self.ready_id.append(env_id)
        obs_list, rew_list, term_list, trunc_list, info_list = result
        term_list = np.array([term_list for _ in range(self.env_num)])
        trunc_list = np.array([trunc_list for _ in range(self.env_num)])
        #info_list = np.array([info_list for _ in range(self.env_num)])
        if not info_list:
            info_list = np.array([info_list for _ in range(self.env_num)])
        try:
            obs_stack = np.stack(obs_list)
        except ValueError:  # different len(obs)
            obs_stack = np.array(obs_list, dtype=object)
        return (
            obs_stack,
            np.stack(rew_list),
            np.stack(term_list),
            np.stack(trunc_list),
            info_list,
        )

    def set_env_attr(
        self,
        key: str,
        value: Any,
        id: int | list[int] | np.ndarray | None = None,
    ) -> None:
        """Set an attribute in the underlying environments.

        If id is an int, set the attribute denoted by key from the environment
        underlying the worker at index id to value.
        Otherwise, set the attribute for all workers at indices id.

        :param str key: The key of the desired attribute.
        :param Any value: The new value of the attribute.
        :param id: Indice(s) of the desired worker(s). Default to None for all env_id.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)
        self.worker.set_env_attr(key, value)
    
    def __len__(self) -> int:
        """Return len(self), which is the number of environments."""
        return self.env.batch_size
    
    def render(self, **kwargs: Any) -> list[Any]:
        """Render all of the environments."""
        self._assert_is_not_closed()
        if self.is_async and len(self.waiting_id) > 0:
            raise RuntimeError(
                f"Environments {self.waiting_id} are still stepping, cannot render them now.",
            )
        return [w.render(**kwargs) for w in self.workers]

    def close(self) -> None:
        """Close all of the environments.

        This function will be called only once (if not, it will be called during
        garbage collected). This way, ``close`` of all workers can be assured.
        """
        self._assert_is_not_closed()
        for w in self.workers:
            w.close()
        self.is_closed = True
    
    def seed(self, seed: int | list[int] | None = None) -> list[list[int] | None]:
        """Set the seed for all environments.

        Accept ``None``, an int (which will extend ``i`` to
        ``[i, i + 1, i + 2, ...]``) or a list.

        :return: The list of seeds used in this env's random number generators.
            The first value in the list should be the "main" seed, or the value
            which a reproducer pass to "seed".
        """
        self._assert_is_not_closed()
        seed_list: list[None] | list[int]
        if seed is None:
            seed_list = [seed] * self.env_num
        elif isinstance(seed, int):
            seed_list = [seed + i for i in range(self.env_num)]
        else:
            seed_list = seed
        return [w.seed(s) for w, s in zip(self.workers, seed_list, strict=True)]
    


    
    
