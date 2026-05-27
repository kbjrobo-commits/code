"""Shared speed model and cushion-shot event rules."""
import numpy as np

from project.config import (
    BALL_SPEED_GAIN,
    MAX_TOOL_SPEED,
)


def predict_ball_speed(tool_speed_cmd, gain=BALL_SPEED_GAIN):
    """Estimate cue-ball initial speed from commanded tool speed."""
    tool_speed = max(float(tool_speed_cmd), 0.0)
    return tool_speed * float(gain)


def tool_speed_for_ball_speed(ball_speed, gain=BALL_SPEED_GAIN):
    """Invert the speed model and clamp to the executable tool-speed limit."""
    gain = max(float(gain), 1e-9)
    return min(max(float(ball_speed) / gain, 0.0), MAX_TOOL_SPEED)


def target_contact_indices(events):
    """Return the first t1/t2 event indices, or (-1, -1) if absent."""
    t1_idx = events.index('t1') if 't1' in events else -1
    t2_idx = events.index('t2') if 't2' in events else -1
    return t1_idx, t2_idx


def valid_cushion_sequence(events, required_cushions):
    """True when both target balls are hit after at least N cushion events.

    The order of target balls is allowed to vary. The cushion count is measured
    before the second target-ball contact, which matches the two/three-cushion
    requirement for this demo.
    """
    t1_idx, t2_idx = target_contact_indices(events)
    if t1_idx < 0 or t2_idx < 0:
        return False
    second_target_idx = max(t1_idx, t2_idx)
    return events[:second_target_idx].count('c') >= required_cushions


def cushion_count_before_second_target(events):
    """Diagnostic cushion count used by the success predicates."""
    t1_idx, t2_idx = target_contact_indices(events)
    if t1_idx < 0 or t2_idx < 0:
        return 0
    return events[:max(t1_idx, t2_idx)].count('c')


class CushionContactTracker:
    """Track cue-ball contacts with target balls and cushion transitions."""

    def __init__(self, target1_id, target2_id, cushion_ids):
        self.target1_id = target1_id
        self.target2_id = target2_id
        self.cushion_ids = set(cushion_ids or [])
        self.events = []
        self.hit_t1 = False
        self.hit_t2 = False
        self.cushion_count = 0
        self._prev_cushions = set()

    def update_from_contacts(self, contacts):
        cur_cushions = set()
        for contact in contacts:
            other_id = contact[2]
            if other_id == self.target1_id and not self.hit_t1:
                self.hit_t1 = True
                self.events.append('t1')
            elif self.target2_id is not None and other_id == self.target2_id and not self.hit_t2:
                self.hit_t2 = True
                self.events.append('t2')
            elif other_id in self.cushion_ids:
                cur_cushions.add(other_id)

        new_cushions = cur_cushions - self._prev_cushions
        for _ in new_cushions:
            self.cushion_count += 1
            self.events.append('c')
        self._prev_cushions = cur_cushions

    @property
    def valid_2cushion(self):
        return valid_cushion_sequence(self.events, 2)

    @property
    def valid_3cushion(self):
        return valid_cushion_sequence(self.events, 3)

    def snapshot(self):
        return {
            'hit_t1': self.hit_t1,
            'hit_t2': self.hit_t2,
            'cushion_count': self.cushion_count,
            'events': list(self.events),
            'valid_2cushion': self.valid_2cushion,
            'valid_3cushion': self.valid_3cushion,
            'cushions_before_second_target': cushion_count_before_second_target(self.events),
        }
