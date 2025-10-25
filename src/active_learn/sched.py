"""
Scheduling utilities for the active-learning loop.
==================================================

Keeps track of iteration counts, triggers retraining events and generator
refresh cadence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["SchedulerConfig", "ActiveLearningScheduler"]


@dataclass
class SchedulerConfig:
    max_iterations: int = 50
    surrogate_retrain_every: int = 1
    generator_refresh_every: int = 5
    log_every: int = 1


@dataclass
class ActiveLearningScheduler:
    config: SchedulerConfig
    iteration: int = 0
    labelled_since_retrain: int = 0
    generated_since_refresh: int = 0
    history: list = field(default_factory=list)

    def step(self, num_labelled: int, num_generated: int) -> None:
        self.iteration += 1
        self.labelled_since_retrain += num_labelled
        self.generated_since_refresh += num_generated
        self.history.append({"iteration": self.iteration, "labelled": num_labelled, "generated": num_generated})

    def should_retrain_surrogate(self) -> bool:
        if self.config.surrogate_retrain_every <= 0:
            return False
        if self.labelled_since_retrain >= self.config.surrogate_retrain_every:
            self.labelled_since_retrain = 0
            return True
        return False

    def should_refresh_generator(self) -> bool:
        if self.config.generator_refresh_every <= 0:
            return False
        if self.generated_since_refresh >= self.config.generator_refresh_every:
            self.generated_since_refresh = 0
            return True
        return False

    def should_stop(self) -> bool:
        return self.iteration >= self.config.max_iterations
