"""새 테이블 배치에서 각도별 도달가능성 + IK 안전도 진단 (v2)"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from project.config import *

CX, CY = MAZE_TABLE_CENTER_X, MAZE_TABLE_CENTER_Y
L, W = MAZE_TABLE_LENGTH, MAZE_TABLE_WIDTH
H = MAZE_TABLE_SURFACE_HEIGHT
TH = MAZE_TABLE_HEIGHT
ball_h = H + TH / 2 + MAZE_BALL_RADIUS + 0.001

cue_pos = np.array([CX, CY - W/4, ball_h])

print(f"=== 테이블 기하학 ===")
print(f"테이블: {L}m(X) x {W}m(Y), center=({CX}, {CY}), h={H}m")
print(f"범위: x=[{CX-L/2:.3f}, {CX+L/2:.3f}], y=[{CY-W/2:.3f}, {CY+W/2:.3f}]")
print(f"Cue: ({cue_pos[0]:.3f}, {cue_pos[1]:.4f}), r={np.linalg.norm(cue_pos[:2]):.3f}m")

bounds = {'x_min': CX - L/2, 'x_max': CX + L/2,
          'y_min': CY - W/2, 'y_max': CY + W/2}

# 1° resolution
reachable_r65 = []
wall_blocked = []
SAFE_RADIUS = 0.65
tip_margin = TOOL_TIP_RADIUS

for deg in range(360):
    angle = np.radians(deg)
    strike_dir = np.array([np.cos(angle), np.sin(angle), 0.0])
    
    if abs(TOOL_YAW_OFFSET) > 1e-6:
        ee_y = np.array([strike_dir[1], -strike_dir[0], 0.0])
        tool_dir = strike_dir * np.cos(TOOL_YAW_OFFSET) + ee_y * np.sin(TOOL_YAW_OFFSET)
        ee_offset = -tool_dir * TOOL_HORIZONTAL_EXT + np.array([0, 0, TOOL_VERTICAL_DROP])
    else:
        ee_offset = -strike_dir * TOOL_HORIZONTAL_EXT + np.array([0, 0, TOOL_VERTICAL_DROP])
    
    ready_pos = cue_pos + ee_offset - strike_dir * STRIKE_APPROACH_DIST
    ready_dist = np.linalg.norm(ready_pos[:2])
    
    # wall check
    sd2 = strike_dir[:2]
    cue2 = cue_pos[:2]
    safe_approach = STRIKE_APPROACH_DIST
    for axis in [0, 1]:
        if abs(sd2[axis]) > 1e-6:
            if sd2[axis] > 0:
                max_a = (cue2[axis] - (bounds['x_min' if axis==0 else 'y_min'] + tip_margin)) / sd2[axis]
            else:
                max_a = (cue2[axis] - (bounds['x_max' if axis==0 else 'y_max'] - tip_margin)) / sd2[axis]
            if max_a > 0:
                safe_approach = min(safe_approach, max_a)
    safe_approach = max(0.08, safe_approach)
    
    tip_check = cue2 - sd2 * safe_approach
    wall_ok = (tip_check[0] >= bounds['x_min'] + tip_margin and
               tip_check[0] <= bounds['x_max'] - tip_margin and
               tip_check[1] >= bounds['y_min'] + tip_margin and
               tip_check[1] <= bounds['y_max'] - tip_margin)
    
    if not wall_ok:
        wall_blocked.append((deg, ready_dist))
    elif ready_dist > SAFE_RADIUS:
        pass  # unreachable
    else:
        reachable_r65.append((deg, ready_dist))

print(f"\n=== 도달 가능 (r<=0.65, no wall): {len(reachable_r65)}/360 ===")

# 그룹별로 정리
close = [(d,r) for d,r in reachable_r65 if r <= 0.50]
mid = [(d,r) for d,r in reachable_r65 if 0.50 < r <= 0.60]
far = [(d,r) for d,r in reachable_r65 if r > 0.60]
print(f"  가까운(r<=0.50): {len(close)} 각도 → manipulability 높음")
if close:
    degs = [d for d,r in close]
    print(f"    각도: {degs[0]}~{degs[-1]}°")
print(f"  중간(0.50<r<=0.60): {len(mid)}")
print(f"  먼(r>0.60): {len(far)} → manipulability 낮을 수 있음")

print(f"\n  벽 뚫림으로 차단: {len(wall_blocked)} 각도")
if wall_blocked:
    degs = [d for d,r in wall_blocked]
    # 연속 구간
    segs = []
    s = degs[0]; p = degs[0]
    for d in degs[1:]:
        if d - p > 1: segs.append((s, p)); s = d
        p = d
    segs.append((s, p))
    print(f"    차단 구간: {segs}")

# 핵심: 현재 angle_priority가 유효한지
print(f"\n=== angle_priority 분석 ===")
print(f"이전 safe(50-180): 이 범위 중 벽 차단 = ", end="")
wall_degs = set(d for d,_ in wall_blocked)
blocked_in_safe = sorted(wall_degs & set(range(50, 181)))
print(f"{blocked_in_safe if blocked_in_safe else '없음'}")

# 어떤 각도가 실제로 가장 가까운(manipulability 좋은) ready pos?
best_degs = sorted(reachable_r65, key=lambda x: x[1])[:20]
print(f"\n  Ready pos 가장 가까운 top-20 각도:")
for d, r in best_degs:
    old_prio = 0 if 50<=d<=180 else (1 if 30<=d<50 or 180<d<=330 else 2)
    print(f"    {d:3d}° → r={r:.3f}m, old_priority={old_prio}")
