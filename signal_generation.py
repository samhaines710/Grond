import numpy as np
import optuna
from collections import defaultdict
from typing import Callable, Dict, Any, List

from movement_classifier import MovementClassifier
from strategy_logic import StrategyLogic

# For reinforcement learning
import gym
from gym import spaces
from stable_baselines3 import PPO


class AdaptiveHyperparamOptimizer:
    """
    Uses Optuna to tune hyperparameters of your trading strategy via backtests.
    """

    def __init__(
        self,
        backtest_func: Callable[[Dict[str, Any]], float],
        param_space: Dict[str, Any],
        n_trials: int = 50,
        direction: str = "maximize"
    ):
        """
        :param backtest_func: function mapping hyperparam dict → performance metric.
        :param param_space: dict of param_name → {"low":…, "high":…, Optional["step"], or "choices":[…]}.
        :param n_trials: number of Optuna trials.
        :param direction: "maximize" or "minimize".
        """
        self.backtest = backtest_func
        self.param_space = param_space
        self.n_trials = n_trials
        self.study = optuna.create_study(direction=direction)

    def _suggest(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = {}
        for name, opts in self.param_space.items():
            if "choices" in opts:
                params[name] = trial.suggest_categorical(name, opts["choices"])
            else:
                low, high = opts["low"], opts["high"]
                step = opts.get("step")
                if step:
                    params[name] = trial.suggest_float(name, low, high, step=step)
                else:
                    params[name] = trial.suggest_float(name, low, high)
        return params

    def optimize(self) -> optuna.Study:
        """
        Run the study and return the Optuna Study object.
        """
        def objective(trial):
            params = self._suggest(trial)
            return self.backtest(params)
        self.study.optimize(objective, n_trials=self.n_trials)
        return self.study


class BanditAllocator:
    """
    Epsilon‐greedy multi‐armed bandit for exploring different movement_types.
    """

    def __init__(self, arms: List[str], epsilon: float = 0.1):
        """
        :param arms: list of discrete options (e.g. movement_type strings).
        :param epsilon: exploration rate (0–1).
        """
        self.epsilon = epsilon
        self.arms = arms
        self.counts = defaultdict(int)
        self.values = defaultdict(float)

    def select_arm(self) -> str:
        """
        Choose an arm: with probability ε choose random; otherwise best historical.
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.arms)
        # compute average reward per arm
        avg = {a: self.values[a] / max(1, self.counts[a]) for a in self.arms}
        # return arm with highest average reward
        return max(avg, key=avg.get)

    def update(self, arm: str, reward: float):
        """
        Update counts and estimated value for the chosen arm.
        """
        self.counts[arm] += 1
        # incremental average update
        self.values[arm] += (reward - self.values[arm] / self.counts[arm])


class TradingEnv(gym.Env):
    """
    OpenAI Gym environment for trading based on MovementClassifier + StrategyLogic.
    """

    metadata = {'render.modes': []}

    def __init__(
        self,
        feature_gen: Callable[[str], Dict[str, Any]],
        strat_logic: StrategyLogic,
        tickers: List[str],
        horizon_steps: int = 3
    ):
        """
        :param feature_gen: function mapping ticker → feature dict.
        :param strat_logic: StrategyLogic instance to execute strategies.
        :param tickers: list of tickers to sample from.
        :param horizon_steps: number of steps per episode.
        """
        super().__init__()
        self.feat = feature_gen
        self.logic = strat_logic
        self.tickers = tickers
        self.horizon = horizon_steps

        # arms are all possible movement_types from logic
        self.arms = list(self.logic.logic_branches.keys())
        self.action_space = spaces.Discrete(len(self.arms))

        # observation dimension determined by feature generator output
        sample = self.feat(self.tickers[0])
        obs_dim = len(sample)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.current_ticker = None
        self.current_step = 0

    def reset(self):
        """
        Start a new episode: pick random ticker and return initial observation.
        """
        self.current_ticker = np.random.choice(self.tickers)
        self.current_step = 0
        feats = self.feat(self.current_ticker)
        return np.array(list(feats.values()), dtype=np.float32)

    def step(self, action: int):
        """
        Take an action (choose movement_type), simulate reward, and return next obs.
        """
        mv = self.arms[action]
        feats = self.feat(self.current_ticker)
        strat = self.logic.execute_strategy(mv, feats)
        reward = self._simulate_pnl(strat, self.current_ticker)

        self.current_step += 1
        done = self.current_step >= self.horizon
        obs = np.array(list(self.feat(self.current_ticker).values()), dtype=np.float32)
        return obs, reward, done, {}

    def _simulate_pnl(self, strat: Dict[str, Any], ticker: str) -> float:
        """
        Placeholder PnL simulator: override with real backtest logic.
        """
        # e.g., use strategy signal and historical data to compute PnL
        return np.random.normal(0, 1)

    def render(self, mode='human'):
        pass


class RLAgent:
    """
    Reinforcement‐learning agent using Stable Baselines3 PPO.
    """

    def __init__(self, env: gym.Env, **ppo_kwargs):
        """
        :param env: a Gym environment.
        :param ppo_kwargs: optional keyword args passed to PPO constructor.
        """
        self.env = env
        self.model = PPO("MlpPolicy", env, verbose=0, **ppo_kwargs)

    def train(self, timesteps: int = 100_000):
        """
        Train the PPO agent for a given number of timesteps.
        """
        self.model.learn(total_timesteps=timesteps)

    def act(self, obs: np.ndarray) -> int:
        """
        Given an observation, return the selected action.
        """
        action, _states = self.model.predict(obs, deterministic=True)
        return int(action)
