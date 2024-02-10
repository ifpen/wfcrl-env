from abc import ABC, abstractmethod


class WindFarmReward(ABC):
    @abstractmethod
    def __call__(self, state, production, action, next_production, next_state):
        pass


class PercentageReward(WindFarmReward):
    def __call__(self, state, production, action, next_production, next_state):
        return (next_production - production) / production
