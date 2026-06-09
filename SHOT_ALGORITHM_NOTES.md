# Shot Algorithm Notes

## 현재 문제 인식

현재 샷 계산은 시간이 오래 걸리는 편이고, 타격 각도도 우리가 원하는 방식과 다르게 동작하는 것으로 보인다.

특히 원하는 동작은 "타겟공이 홀에 들어가는 경로"를 기준으로 샷을 계산하는 것인데, 현재는 "흰공이 홀 쪽으로 들어가는 경로" 또는 흰공 기준의 단순 타격 각도에 가깝게 계산되는 느낌이 있다.

우리가 원하는 샷은 다음 순서로 계산되어야 한다.

1. 타겟공이 어떤 홀로 들어갈지 결정한다.
2. 타겟공이 그 홀로 굴러가기 위해 필요한 진행 방향을 계산한다.
3. 흰공이 타겟공의 어느 지점을 맞춰야 타겟공이 그 방향으로 움직이는지 계산한다.
4. 그 접촉 지점을 맞추기 위해 큐/툴이 어떤 위치와 각도를 가져야 하는지 계산한다.
5. 계산된 툴 pose를 이용해 로봇이 타격 모션을 수행한다.

## 원하는 알고리즘 흐름

### 1. 타겟공과 홀 선택

각 타겟공과 각 홀 조합에 대해 가능한 샷인지 검사한다.

- 타겟공에서 홀까지 직선 경로가 막혀 있지 않은지 확인한다.
- 흰공에서 타겟공의 접촉 지점까지 경로가 막혀 있지 않은지 확인한다.
- 타격 각도가 너무 얇거나 불가능한 각도인지 확인한다.
- 가능한 후보 중 거리, 각도, 장애물 여부 등을 기준으로 가장 좋은 샷을 선택한다.

### 2. 타겟공이 홀로 들어가는 방향 계산

타겟공 위치를 `target`, 홀 위치를 `pocket`이라고 하면, 타겟공이 가야 하는 방향은 다음 벡터다.

```text
target_to_pocket = normalize(pocket - target)
```

이 방향은 "타겟공이 충돌 이후 움직여야 하는 방향"이다.

### 3. 흰공이 맞춰야 하는 타겟공의 접촉 지점 계산

두 공의 충돌을 단순화하면, 흰공의 중심은 충돌 순간에 타겟공 중심에서 `공 지름`만큼 뒤쪽에 있어야 한다.

즉 타겟공을 홀 방향으로 보내려면, 흰공의 중심이 도달해야 하는 ghost ball 위치는 다음과 같다.

```text
ghost_ball = target - target_to_pocket * ball_diameter
```

여기서 `ghost_ball`은 실제 공이 아니라, 흰공이 충돌 순간에 있어야 하는 중심 위치다.

정리하면:

- 타겟공은 `target_to_pocket` 방향으로 움직여야 한다.
- 흰공은 `ghost_ball` 위치를 향해 움직여야 한다.
- 따라서 큐의 타격 방향은 `cue_ball -> ghost_ball` 방향이다.

### 4. 큐/툴 타격 방향 계산

흰공 위치를 `cue_ball`이라고 하면 큐가 향해야 하는 방향은 다음과 같다.

```text
cue_to_ghost = normalize(ghost_ball - cue_ball)
strike_angle = atan2(cue_to_ghost.y, cue_to_ghost.x)
```

로봇 툴의 위치는 흰공 중심에서 타격 방향의 반대쪽으로 일정 거리만큼 떨어진 곳에 둔다.

```text
tool_position = cue_ball - cue_to_ghost * cue_offset
```

현재 코드에서는 이 `cue_offset`에 해당하는 값으로 `self.d = 0.12`를 사용하고 있다.

### 5. 로봇 pose 생성

최종적으로 로봇에게 넘길 pose는 다음 정보를 가져야 한다.

- `position.x`, `position.y`: 흰공 뒤쪽의 툴 위치
- `position.z`: 큐가 공을 칠 수 있는 높이
- `orientation`: 큐가 `strike_angle` 방향을 바라보도록 하는 회전값

현재 코드에서는 `calc_strike_pose()`에서 다음과 같이 orientation을 만든다.

```python
q = quaternion_from_euler(np.pi, 0.0, float(strike_ang) - np.pi / 4)
```

이 부분은 실제 gripper, cue stick, tool frame 축 방향에 맞춘 보정값으로 보인다. 따라서 알고리즘을 수정할 때도 `strike_ang` 자체는 위의 ghost ball 방식으로 계산하되, 최종 quaternion 보정은 기존 툴 프레임 정의를 유지하면서 검증하는 것이 좋다.

## 현재 코드와 비교

`poolAlgorithm.py`의 `calc_cue_pos()`는 이미 구조상으로는 아래 흐름을 일부 갖고 있다.

```text
target -> pocket 각도 계산
target 뒤쪽의 cue final 위치 계산
cue_ball -> cue final 방향 계산
tool 위치 계산
```

하지만 실제 동작이 "흰공이 홀로 들어가는 방향"처럼 보인다면 다음을 의심해볼 수 있다.

- cue ball과 target ball 이름이 실제 시스템에서 올바르게 들어가고 있는지
- `red_ball`이 cue ball로 고정되어 있는데, 알고리즘 내부 일부는 `blue_ball`을 cue ball처럼 가정하고 있지 않은지
- `cf` 또는 ghost ball 위치 계산이 공 반지름/지름 기준으로 맞는지
- 툴 프레임 orientation 보정값 `strike_ang - np.pi / 4`가 실제 큐 방향과 일치하는지
- `world.strikeTransform()` 이후 실제 타격 방향이 의도한 방향과 같은지
- 후보 샷 선택 시 첫 번째 가능한 공/홀 조합을 바로 반환해서 좋지 않은 샷이 선택되고 있지 않은지

## 개선 방향

### 후보 샷 스코어링 추가

현재처럼 가능한 샷을 찾자마자 반환하면, 이상한 후보가 먼저 선택될 수 있다. 모든 타겟공-홀 후보를 계산한 뒤 점수를 매겨 가장 좋은 샷을 선택하는 방식이 좋다.

예시 점수 요소:

- 타겟공에서 홀까지의 거리
- 흰공에서 ghost ball까지의 거리
- cut angle의 크기
- 장애물 여부
- 테이블 벽과 너무 가까운지 여부
- 로봇이 도달하기 쉬운 pose인지 여부

### ghost ball 계산을 명시적으로 분리

계산을 함수로 나누면 디버깅이 쉬워진다.

```python
def calc_ghost_ball(target_ball, pocket):
    direction = normalize(pocket - target_ball)
    ghost = target_ball - direction * D_ball
    return ghost
```

그리고 타격 각도 계산도 별도 함수로 분리한다.

```python
def calc_strike_angle(cue_ball, ghost_ball):
    return atan2(ghost_ball.y - cue_ball.y, ghost_ball.x - cue_ball.x)
```

### 디버그 로그/시각화 추가

샷이 이상하게 나갈 때는 숫자만 보면 원인 파악이 어렵다. 다음 값을 로그나 RViz marker로 표시하면 좋다.

- 선택된 타겟공
- 선택된 홀
- target-to-pocket 방향
- ghost ball 위치
- cue-to-ghost 방향
- 최종 tool position
- 최종 strike angle

## 결론

우리가 원하는 알고리즘은 "흰공을 홀로 보내는 각도"가 아니라, "타겟공을 홀로 보내기 위한 ghost ball 위치를 계산하고, 흰공이 그 위치를 향하도록 큐/툴 pose를 만드는 방식"이어야 한다.

핵심 공식은 다음 두 개다.

```text
ghost_ball = target_ball - normalize(pocket - target_ball) * ball_diameter
strike_angle = atan2(ghost_ball.y - cue_ball.y, ghost_ball.x - cue_ball.x)
```

이 계산을 기준으로 툴 위치와 orientation을 만들고, 그 pose를 현재 `strike_ball()`의 3단계 타격 모션에 넘기면 된다.
