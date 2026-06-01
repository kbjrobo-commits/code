"""
사이드 포켓(홀) 회피 — 기하학적 1쿠션 예측 !!
==========================================
가정: 포켓볼 사이드 포켓 2개 = 테이블 **긴 변(y_min / y_max) 레일 정중앙**
      (레일 방향 X, 개구부 ~5cm). PyBullet에는 구멍 없음 — 탐색 필터만.
"""
import numpy as np

from project.config import (
    MAZE_SIDE_POCKET_AVOID,
    MAZE_SIDE_POCKET_HALF_LENGTH,
    MAZE_SIDE_POCKET_INWARD_DEPTH,
    MAZE_SIDE_POCKET_MARGIN,
    MAZE_TABLE_LENGTH,
    MAZE_TABLE_WIDTH,
)


def long_rail_pocket_center_x(table_bounds):
    """긴 변 레일(X 방향) 중앙 X — 사이드 포켓 위치."""
    return 0.5 * (table_bounds['x_min'] + table_bounds['x_max'])


def long_rail_side_pocket_centers(table_bounds):
    """긴 변 2곳 사이드 포켓 중심 (x, y) — 로그/디버그용."""
    cx = long_rail_pocket_center_x(table_bounds)
    return [
        (cx, table_bounds['y_max']),
        (cx, table_bounds['y_min']),
    ]


def side_pocket_forbidden_zones(table_bounds):
    """긴 변(y±) 중앙 사이드 포켓 회피 구역 (테이블 내부 AABB)."""
    cx = long_rail_pocket_center_x(table_bounds)
    half = MAZE_SIDE_POCKET_HALF_LENGTH + MAZE_SIDE_POCKET_MARGIN
    depth = MAZE_SIDE_POCKET_INWARD_DEPTH
    y_min = table_bounds['y_min']
    y_max = table_bounds['y_max']
    return [
        {
            'x_min': cx - half,
            'x_max': cx + half,
            'y_min': y_max - depth,
            'y_max': y_max,
            'rail': 'y_max',
            'center': (cx, y_max),
        },
        {
            'x_min': cx - half,
            'x_max': cx + half,
            'y_min': y_min,
            'y_max': y_min + depth,
            'rail': 'y_min',
            'center': (cx, y_min),
        },
    ]


def _playable_bounds(table_bounds, ball_r):
    return {
        'x_min': table_bounds['x_min'] + ball_r,
        'x_max': table_bounds['x_max'] - ball_r,
        'y_min': table_bounds['y_min'] + ball_r,
        'y_max': table_bounds['y_max'] - ball_r,
    }


def ray_first_cushion(pos2d, direction2d, table_bounds, ball_r):
    """공 중심 궤적의 첫 쿠션: (hit, wall, pos_after, dir_after) 또는 None."""
    b = _playable_bounds(table_bounds, ball_r)
    px, py = float(pos2d[0]), float(pos2d[1])
    dx, dy = np.asarray(direction2d[:2], dtype=float).flatten()
    norm = np.linalg.norm([dx, dy])
    if norm < 1e-9:
        return None
    dx /= norm
    dy /= norm

    t_best = float('inf')
    wall = None

    if dx > 1e-9:
        t = (b['x_max'] - px) / dx
        if 1e-9 < t < t_best:
            t_best, wall = t, 'x_max'
    elif dx < -1e-9:
        t = (b['x_min'] - px) / dx
        if 1e-9 < t < t_best:
            t_best, wall = t, 'x_min'

    if dy > 1e-9:
        t = (b['y_max'] - py) / dy
        if 1e-9 < t < t_best:
            t_best, wall = t, 'y_max'
    elif dy < -1e-9:
        t = (b['y_min'] - py) / dy
        if 1e-9 < t < t_best:
            t_best, wall = t, 'y_min'

    if wall is None:
        return None

    hit = np.array([px + dx * t_best, py + dy * t_best])
    rdx, rdy = (-dx if wall in ('x_min', 'x_max') else dx,
                (-dy if wall in ('y_min', 'y_max') else dy))
    rdir = np.array([rdx, rdy])
    rdir /= np.linalg.norm(rdir)

    eps = max(ball_r * 0.25, 1e-4)
    if wall == 'x_max':
        pos_after = np.array([b['x_max'] - eps, hit[1]])
    elif wall == 'x_min':
        pos_after = np.array([b['x_min'] + eps, hit[1]])
    elif wall == 'y_max':
        pos_after = np.array([hit[0], b['y_max'] - eps])
    else:
        pos_after = np.array([hit[0], b['y_min'] + eps])

    return hit, wall, pos_after, rdir


def _ray_hits_aabb(origin, direction, box, max_dist):
    """반직선 origin + t*direction (t>=0, t<=max_dist) 이 AABB 와 교차하는지."""
    tmin, tmax = 0.0, float(max_dist)
    o = np.asarray(origin[:2], dtype=float)
    d = np.asarray(direction[:2], dtype=float)
    for i, (bmin, bmax) in enumerate(
        [(box['x_min'], box['x_max']), (box['y_min'], box['y_max'])]
    ):
        if abs(d[i]) < 1e-12:
            if o[i] < bmin or o[i] > bmax:
                return False
            continue
        t1 = (bmin - o[i]) / d[i]
        t2 = (bmax - o[i]) / d[i]
        if t1 > t2:
            t1, t2 = t2, t1
        tmin = max(tmin, t1)
        tmax = min(tmax, t2)
        if tmin > tmax:
            return False
    return tmax >= 0.0


def _hit_on_side_pocket_segment(hit_x, wall, table_bounds):
    """첫 쿠션이 긴 변(y±) 중앙 포켓 구간에 닿는지."""
    if wall not in ('y_min', 'y_max'):
        return False
    cx = long_rail_pocket_center_x(table_bounds)
    half = MAZE_SIDE_POCKET_HALF_LENGTH + MAZE_SIDE_POCKET_MARGIN
    return abs(float(hit_x) - cx) <= half


def rejects_first_cushion_toward_side_pocket(cue_pos2d, angle_rad, table_bounds, ball_r):
    """첫 쿠션 후 긴 변 중앙 사이드 포켓 쪽 경로면 True (탐색 제외)."""
    if not MAZE_SIDE_POCKET_AVOID:
        return False

    bounce = ray_first_cushion(cue_pos2d, [np.cos(angle_rad), np.sin(angle_rad)],
                               table_bounds, ball_r)
    if bounce is None:
        return False

    hit, wall, pos_after, dir_after = bounce
    zones = side_pocket_forbidden_zones(table_bounds)

    if _hit_on_side_pocket_segment(hit[0], wall, table_bounds):
        return True

    max_travel = MAZE_TABLE_LENGTH + MAZE_TABLE_WIDTH
    for zone in zones:
        if _ray_hits_aabb(pos_after, dir_after, zone, max_travel):
            return True

    return False
