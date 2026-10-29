"""Explicit, bounded policy for the opt-in layered quality loop."""
from dataclasses import dataclass


@dataclass(frozen=True)
class LoopPolicy:
    max_extra_vision_calls: int = 2
    max_repair_rounds: int = 2

    def __post_init__(self):
        for name in ('max_extra_vision_calls', 'max_repair_rounds'):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= 2:
                raise ValueError(f'{name} must be an integer from 0 to 2')
