from abc import ABC, abstractmethod


class RewardShaper(ABC):
    @abstractmethod
    def __call__(self, reward: float):
        pass

    def update(self):
        pass


class ReferencePercentage(RewardShaper):
    def __init__(self, reference: float):
        self.reference = reference

    def __call__(self, reward):
        return (reward - self.reference) / self.reference


class StepPercentage(RewardShaper):
    def __init__(self, reference: float = 0.0):
        self.reference = reference

    def __call__(self, reward):
        if self.reference == 0:
            shaped_reward = 0.0
        else:
            shaped_reward = (reward - self.reference) / self.reference
        self.reference = reward
        return shaped_reward
