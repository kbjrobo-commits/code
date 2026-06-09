# Indy7 포켓볼 최종 데모 전략

## 0. 최종 프로젝트 메시지

본 프로젝트의 핵심은 포켓볼대를 로봇에 맞게 바꾸는 것이 아니라, **기존 포켓볼대 환경을 그대로 유지한 상태에서 로봇이 현재 공 배치와 물리 오차를 인식하고, 최소한의 시도로 task를 성공시키는 것**이다.

최종 메시지:

> 환경을 로봇에 맞추는 것이 아니라, 로봇이 현재 환경에 맞춰 포켓볼 task를 수행한다.

또는:

> 단순히 공을 치는 로봇이 아니라, 현재 공 배치를 인식하고 가장 성공 가능성이 높은 다음 샷을 선택하며, 실패 결과를 이용해 보정하는 로봇팔 포켓볼 시스템을 구현한다.

---

## 1. 최종 데모 구성

### Demo 1. 최소 시도 포켓팅

목표:

> 현재 테이블 위의 여러 목적구를 가능한 적은 시도 횟수로 6개 포켓 중 하나에 넣는다.

핵심 흐름:

```text
공/포켓 인식
→ 모든 공-포켓 후보 생성
→ 각 후보 shot score 계산
→ 가장 성공 가능성이 높은 shot 선택
→ ghost ball 기반 타격 방향 계산
→ Indy7 실행 가능성 확인
→ 로봇 타격
→ 공 위치 재인식
→ 다음 shot 재계획
→ 모든 공 포켓팅
```

현실적인 목표:

- 목적구 2~3개부터 시작한다.
- 모든 공을 완전 자동으로 넣는 것보다, **각 shot마다 후보를 평가하고 가장 쉬운 shot을 선택하는 과정**을 보여주는 것이 중요하다.
- 성공 기준은 “모든 공을 N회 이내에 포켓팅” 또는 “선택된 shot의 성공/실패 후 재계획 수행”으로 잡는다.

권장 성공 조건:

| 수준 | 목적구 수 | 성공 기준 |
|---|---:|---|
| Level 1 | 1개 | 지정 공을 6개 포켓 후보 중 최적 포켓에 넣기 |
| Level 2 | 2개 | 두 공을 순차적으로 포켓팅 |
| Level 3 | 3개 | 세 공을 greedy planner로 순차 포켓팅 |
| Challenge | 4개 이상 | break 후 쉬운 공부터 포켓팅 |

---

### Demo 2. 트릭샷

목표:

> 단일 포켓팅을 넘어, 한 번의 타격으로 두 개 이상의 공에 영향을 주는 물리 task를 보여준다.

추천 트릭샷 후보:

#### A. 한 번의 타격으로 두 공 모두 맞히기

```text
Cue ball → Object ball A → Object ball B
```

성공 조건:

- 흰 공 또는 목적구 A가 목적구 B까지 연쇄적으로 접촉하면 성공.
- 가장 안정적이고 구현 가능성이 높다.

#### B. 두 공을 각각 target zone으로 보내기

```text
Cue ball → Object ball A, Object ball B
A → Target zone A
B → Target zone B
```

성공 조건:

- 두 목적구가 각각 지정된 target zone 근처로 이동하면 성공.
- target zone은 처음에는 반지름 8~12 cm 정도로 크게 잡는다.

#### C. 장애물 사이 통과 후 포켓팅 또는 target zone 도달

```text
Cue ball → obstacle gap → Object ball 또는 Target zone
```

성공 조건:

- 공이 장애물 사이를 통과하고 목표 영역 또는 목적구에 도달하면 성공.
- path planning과 collision avoidance를 보여주기 좋다.

---

## 2. 이 task에 필요한 카메라 수

### 결론

최소 구성은 **카메라 1대**로 가능하다.

가장 추천하는 구성은:

> **상부 고정 카메라 1대 + 필요 시 보조 카메라 1대**

---

### 2.1 최소 구성: 상부 고정 카메라 1대

포켓볼 task는 대부분 2D 평면 상의 문제이므로, 당구대 위에서 내려다보는 카메라 1대만으로도 다음 정보를 얻을 수 있다.

필요한 인식 대상:

- 흰 공 위치
- 목적구 위치
- 6개 포켓 위치
- 테이블 경계
- target zone 위치
- 장애물 또는 다른 공 위치

가능한 기능:

- 공 중심 검출
- 포켓 좌표 등록
- homography 기반 pixel-to-table 좌표 변환
- ghost ball 위치 계산
- 공-포켓 후보 평가
- 타격 후 결과 위치 측정
- 보정 전/후 오차 계산

1대 카메라 조건:

| 조건 | 설명 |
|---|---|
| 카메라 위치 | 테이블 중앙 위쪽 고정 |
| 시야 | 6개 포켓과 모든 공이 들어와야 함 |
| 캘리브레이션 | 테이블 네 모서리 또는 AprilTag/ArUco marker 사용 |
| 좌표 변환 | Homography로 image plane → table plane 변환 |
| 높이 정보 | 필요 없음. 공은 2D 평면에서 추적 가능 |

발표용 표현:

> 포켓볼 task는 테이블 평면에서의 2D 공 위치와 포켓 위치가 핵심이므로, 상부 고정 카메라 1대를 이용해 공과 포켓을 인식하고 homography를 통해 로봇 좌표계로 변환한다.

---

### 2.2 권장 구성: 카메라 2대

가능하면 카메라 2대가 더 안정적이다.

구성:

1. 상부 고정 카메라
   - 공/포켓/테이블 좌표 인식
   - shot planning용 메인 카메라

2. 보조 카메라
   - 로봇 툴팁과 흰 공 정렬 확인
   - 타격 직전 contact point 확인
   - 툴팁 높이, 충돌 위험, 공 중심 정렬 확인

카메라 2대의 장점:

| 역할 | 장점 |
|---|---|
| 상부 카메라 | 전체 공 배치와 포켓 후보 평가 가능 |
| 보조 카메라 | 타격 직전 툴팁-공 정렬 오차 확인 가능 |
| 두 카메라 조합 | perception과 execution을 분리해 안정성 증가 |

하지만 시간과 구현 난이도를 고려하면, 최종 데모는 **상부 카메라 1대 기반**으로 잡는 것이 현실적이다.

---

### 2.3 카메라 수 최종 추천

| 구성 | 추천도 | 설명 |
|---|---:|---|
| 상부 카메라 1대 | ★★★★★ | 최소 구현 가능. 최종 데모에 가장 현실적 |
| 상부 1대 + 측면 1대 | ★★★★☆ | 툴팁 정렬 검증까지 가능. 시간이 되면 추가 |
| 로봇 손목 카메라 | ★★☆☆☆ | 근접 정렬에는 좋지만 전체 공 배치 인식에는 불리 |
| 여러 대 카메라 | ★★☆☆☆ | 정확도는 좋아지지만 캘리브레이션 부담 증가 |

최종 선택:

> **1대 상부 고정 카메라를 기본으로 사용하고, 툴팁 정렬이 불안정할 때만 보조 카메라 또는 수동 정렬 체크를 추가한다.**

---

## 3. 필요한 핵심 기술

### 3.1 Perception

필요 기능:

- 테이블 영역 검출
- 공 중심 검출
- 공 색상/번호 구분 또는 수동 ID 지정
- 포켓 위치 등록
- 장애물 공 위치 검출
- 타격 후 공 정지 위치 측정

추천 구현:

- OpenCV 사용
- 색상 기반 segmentation 또는 Hough Circle Transform
- 공 ID 구분이 어려우면 초기에는 사람이 공 ID를 선택하거나 색상 스티커 사용
- 포켓 위치는 처음 한 번 수동 등록해도 충분함

필요 코드 파일 예시:

```text
vision/ball_detector.py
vision/table_calibration.py
vision/pocket_registration.py
vision/state_estimator.py
```

---

### 3.2 Coordinate Calibration

필요 기능:

- 이미지 좌표계에서 테이블 좌표계로 변환
- 테이블 좌표계에서 로봇 base 좌표계로 변환
- 공 중심, 포켓, target zone 좌표 변환

핵심 기술:

- Homography calibration
- pixel-to-meter scale 변환
- rigid transform 또는 hand-eye calibration의 간단 버전

추천 방식:

1. 테이블 네 모서리 또는 마커 4개를 이미지에서 선택
2. 실제 테이블 좌표를 입력
3. OpenCV `findHomography`로 변환 행렬 계산
4. 로봇 base 기준 테이블 원점과 x-y 방향을 수동 측정
5. table frame → robot base frame 변환 적용

필요 코드 파일 예시:

```text
calibration/homography_calibration.py
calibration/table_to_robot_transform.py
calibration/calibration_utils.py
```

---

### 3.3 Shot Planning

필요 기능:

- 공 N개와 포켓 6개 조합 생성
- 각 공-포켓 후보에 대해 ghost ball 계산
- cue ball에서 ghost ball까지의 타격 방향 계산
- cut angle 계산
- 장애물 충돌 여부 판단
- 후보 shot score 계산
- 최고 score shot 선택

기본 후보 수:

```text
후보 shot 수 = 목적구 개수 N × 포켓 6개
```

예:

```text
목적구 3개 → 3 × 6 = 18개 후보
```

필요 코드 파일 예시:

```text
planning/ghost_ball.py
planning/candidate_generator.py
planning/shot_scorer.py
planning/greedy_planner.py
planning/collision_checker.py
```

---

### 3.4 Ghost Ball 계산

공 위치:

```text
C = cue ball position
O = object ball position
P = target pocket position
r = ball radius
```

object ball이 pocket 방향으로 가야 하므로:

```text
u = (P - O) / ||P - O||
ghost_ball = O - 2r * u
strike_direction = ghost_ball - C
```

의미:

- object ball이 포켓 방향으로 진행하려면 cue ball 중심이 충돌 순간 ghost ball 위치에 와야 한다.
- 로봇은 cue ball을 ghost ball 방향으로 타격한다.

---

### 3.5 Shot Scoring

후보 shot마다 점수를 매긴다.

추천 score 항목:

| 항목 | 의미 |
|---|---|
| Pocketing angle score | cut angle이 작을수록 좋음 |
| Distance score | 목적구-포켓 거리와 cue ball-ghost ball 거리가 적절할수록 좋음 |
| Collision-free score | 경로 중간에 다른 공이 없을수록 좋음 |
| Robot feasibility score | Indy7이 실제로 칠 수 있는 방향일수록 좋음 |
| Speed feasibility score | 필요한 속도가 가능한 범위일수록 좋음 |
| Empirical correction score | 이전 실험에서 성공률이 높았던 조건이면 가산점 |

예시 score:

```text
Shot Score =
0.25 × pocketing_angle_score
+ 0.20 × distance_score
+ 0.20 × collision_free_score
+ 0.20 × robot_feasibility_score
+ 0.10 × speed_feasibility_score
+ 0.05 × empirical_score
```

주의:

- 1주일 안에 완벽한 score를 구현할 필요는 없다.
- 처음에는 angle, distance, collision, robot feasibility만 사용해도 충분하다.

---

### 3.6 Robot Feasibility Check

필요 기능:

- 타격 위치에 로봇이 접근 가능한지 확인
- ㄴ자형 툴팁이 수평 방향 impulse를 전달할 수 있는지 확인
- 로봇 관절 제한과 작업공간 확인
- 당구대/공/포켓과 충돌하지 않는 접근 경로 확인
- 필요한 TCP 속도가 로봇 안전 범위 안인지 확인

검사 항목:

| 검사 | 설명 |
|---|---|
| IK 가능 여부 | 타격 자세를 만들 수 있는가 |
| 접근 방향 | 툴팁이 cue ball 뒤쪽에서 접근 가능한가 |
| 작업공간 | 인디7 reach 안에 있는가 |
| 충돌 | 로봇 링크나 툴이 테이블과 충돌하지 않는가 |
| TCP speed | 필요한 속도가 안전한 범위인가 |

필요 코드 파일 예시:

```text
robot/ik_checker.py
robot/trajectory_generator.py
robot/indy7_controller.py
robot/safety_checker.py
```

---

### 3.7 Robot Execution

추천 실행 구간:

1. Pre-approach
   - 공 뒤쪽 안전 위치로 이동
2. Alignment
   - 타격 방향과 툴팁 방향 정렬
3. Strike
   - 짧은 직선 구간을 moveL 또는 속도 제어로 타격
4. Retreat
   - 타격 후 공과 테이블에서 빠져나오기

실행 방식:

- 접근: 안정적인 moveJ 또는 moveL
- 타격: 짧은 moveL 또는 속도 제어
- 타격 속도: break shot은 높게, 포켓팅은 중간 속도, target shot은 보정 속도 사용

필요 코드 파일 예시:

```text
robot/motion_primitives.py
robot/strike_executor.py
robot/indy7_api_wrapper.py
```

---

### 3.8 Adaptive Correction

필요 기능:

- 타격 전 예측 결과 저장
- 타격 후 실제 결과 측정
- 예측 결과와 실제 결과 비교
- 다음 shot의 각도 또는 속도 보정

보정 대상:

| 보정 항목 | 적용 task |
|---|---|
| 타격 방향 보정 | 포켓팅, target circle, 트릭샷 |
| 타격 속도 보정 | target circle, 두 공 위치 제어 |
| ghost ball offset 보정 | 지정 공-지정 포켓 포켓팅 |
| 경험 기반 score 보정 | 최소 시도 포켓팅 |

간단 보정식:

```text
θ_next = θ_current + kθ × lateral_error
v_next = v_current + kv × distance_error
```

포켓팅 보정:

```text
object ball이 포켓 왼쪽으로 빗나감
→ ghost ball 위치를 오른쪽 방향으로 보정

object ball이 포켓 오른쪽으로 빗나감
→ ghost ball 위치를 왼쪽 방향으로 보정
```

필요 코드 파일 예시:

```text
learning/error_logger.py
learning/shot_correction.py
learning/empirical_score_updater.py
```

---

## 4. 전체 코드 구조 제안

```text
indy7_billiards/
│
├── main.py
│
├── config/
│   ├── camera_config.yaml
│   ├── table_config.yaml
│   ├── robot_config.yaml
│   └── planner_weights.yaml
│
├── vision/
│   ├── ball_detector.py
│   ├── table_detector.py
│   ├── pocket_registration.py
│   ├── marker_detector.py
│   └── state_estimator.py
│
├── calibration/
│   ├── homography_calibration.py
│   ├── table_to_robot_transform.py
│   └── calibration_utils.py
│
├── planning/
│   ├── ghost_ball.py
│   ├── candidate_generator.py
│   ├── collision_checker.py
│   ├── shot_scorer.py
│   └── greedy_planner.py
│
├── robot/
│   ├── indy7_api_wrapper.py
│   ├── ik_checker.py
│   ├── motion_primitives.py
│   ├── trajectory_generator.py
│   ├── strike_executor.py
│   └── safety_checker.py
│
├── learning/
│   ├── error_logger.py
│   ├── shot_correction.py
│   └── empirical_score_updater.py
│
├── demo/
│   ├── demo_minimum_shot_pocketing.py
│   ├── demo_target_circle.py
│   └── demo_trick_shot.py
│
└── logs/
    ├── shot_history.csv
    └── calibration_history.csv
```

---

## 5. 주요 코드별 역할

### main.py

역할:

- 전체 시스템 실행
- 카메라 인식 → planning → robot execution → result logging loop 실행

---

### vision/ball_detector.py

역할:

- 이미지에서 공 중심 검출
- 공 색상 또는 ID 추정
- cue ball과 object ball 구분

필요 출력:

```python
balls = [
    {"id": "cue", "x": 0.12, "y": 0.34},
    {"id": "ball_1", "x": 0.42, "y": 0.21},
]
```

---

### calibration/homography_calibration.py

역할:

- 이미지 좌표를 테이블 좌표로 변환
- 테이블 네 모서리 또는 marker 기준으로 homography 계산

---

### planning/candidate_generator.py

역할:

- 모든 object ball과 6개 pocket 조합 생성

예:

```python
candidates = generate_candidates(object_balls, pockets)
```

---

### planning/ghost_ball.py

역할:

- object ball과 target pocket으로부터 ghost ball 위치 계산
- cue ball이 향해야 할 strike direction 계산

---

### planning/collision_checker.py

역할:

- cue ball → ghost ball 경로에 다른 공이 있는지 확인
- object ball → pocket 경로에 다른 공이 있는지 확인

---

### planning/shot_scorer.py

역할:

- 각 후보 shot의 난이도 점수 계산
- angle, distance, collision, robot feasibility, speed feasibility 반영

---

### planning/greedy_planner.py

역할:

- 현재 상태에서 가장 score가 높은 shot 선택
- shot이 성공 또는 실패한 뒤 재계획

---

### robot/indy7_controller.py 또는 indy7_api_wrapper.py

역할:

- 인디7 API 호출 래핑
- moveJ, moveL, teleop, stop, status check 등 관리

---

### robot/strike_executor.py

역할:

- 타격 자세 생성
- 접근-정렬-타격-후퇴 motion 실행

---

### learning/error_logger.py

역할:

- shot 전 예측 정보 저장
- shot 후 실제 결과 저장
- trial별 오차 기록

---

### learning/shot_correction.py

역할:

- 포켓 실패 방향 또는 target error 기반으로 다음 shot 보정
- ghost ball offset, strike angle, strike speed 업데이트

---

## 6. 최소 구현 버전

시간이 부족할 경우 모든 코드를 완성할 필요는 없다.

### 최소 구현 목표

```text
1. 상부 카메라로 공 위치 인식 또는 수동 좌표 입력
2. 6개 포켓 좌표는 수동 등록
3. 목적구 2~3개에 대해 공-포켓 후보 생성
4. ghost ball 계산
5. 간단한 score로 최고 후보 선택
6. 로봇이 해당 방향으로 타격
7. 타격 후 공 위치 재인식
8. 다음 shot 재계획
```

### 최소 score 항목

```text
1. cut angle
2. 목적구-포켓 거리
3. 경로 collision 여부
4. 로봇 접근 가능 여부
```

### 1주일 내 우선순위

| 우선순위 | 구현 요소 | 이유 |
|---:|---|---|
| 1 | 상부 카메라 좌표계 캘리브레이션 | 모든 planning의 기반 |
| 2 | 공/포켓 좌표 검출 또는 수동 등록 | 후보 shot 생성에 필요 |
| 3 | ghost ball 계산 | 포켓팅 task의 핵심 |
| 4 | 후보 shot score 계산 | 최소 시도 전략의 핵심 |
| 5 | 인디7 타격 motion 안정화 | 실제 데모 성공에 필수 |
| 6 | 타격 후 재인식 | closed-loop 구조 증명 |
| 7 | 간단한 보정식 적용 | sim-to-real 적응성 증명 |
| 8 | 트릭샷 구성 | 발표 임팩트 강화 |

---

## 7. 교수님을 만족시킬 수 있는 핵심 포인트

### 7.1 기존 환경을 바꾸지 않는다

교수님이 우려하는 포인트는 “환경을 쉽게 만들어서 성공률을 높이는 것”일 수 있다.

따라서 최종 발표에서는 다음 메시지를 강조한다.

> 기존 포켓볼대 표면과 쿠션을 그대로 사용하고, 로봇이 현재 환경에서 발생하는 오차를 측정하고 보정하는 방향으로 접근하였다.

---

### 7.2 한 번의 성공 장면보다 시스템 구조를 보여준다

단순 성공 영상 하나보다 중요한 것은 다음 loop이다.

```text
인식 → 후보 생성 → score 평가 → shot 선택 → 실행 → 재인식 → 보정/재계획
```

교수님이 만족할 가능성이 높은 이유:

- 로보틱스개론의 좌표계 변환, 기구학, 경로 계획, feedback 개념이 모두 들어간다.
- 단순 기계 장치가 아니라 robot system으로 보인다.

---

### 7.3 6개 포켓을 모두 고려한 greedy shot planner

포켓이 6개이므로, 로봇은 목적구마다 6개 포켓 후보를 평가할 수 있다.

강조 문장:

> 로봇은 특정 포켓 하나만 목표로 삼는 것이 아니라, 현재 공 배치에서 모든 공-포켓 조합을 생성하고, 가장 성공 가능성이 높은 shot을 선택하여 전체 시도 횟수를 줄이는 greedy planner를 사용한다.

---

### 7.4 Robot-aware planning

단순한 당구 경로 계산이 아니라, 인디7이 실제로 실행 가능한 shot만 선택한다.

검토 항목:

- IK 가능성
- 작업공간
- ㄴ자형 툴팁 접근 방향
- TCP 속도 제한
- 테이블 충돌 가능성

강조 문장:

> 이론적으로 가능한 샷이 아니라, 인디7이 실제로 칠 수 있는 샷을 선택한다.

---

### 7.5 Closed-loop re-planning

한 번 친 뒤 공 위치는 바뀐다. 따라서 다음 shot은 새로 계산해야 한다.

강조 문장:

> 매 타격 후 공의 위치를 다시 인식하고, 변화한 상태에서 다음 shot을 재계획한다.

---

### 7.6 Sim-to-real 보정

기존 환경을 바꾸지 않는 대신, 실제 결과를 보고 보정한다.

보여줄 수 있는 자료:

| 자료 | 의미 |
|---|---|
| Trial별 목표 오차 | 로봇이 점점 목표에 가까워지는지 확인 |
| 보정 전/후 성공률 | 보정 효과 정량화 |
| 예측 경로 vs 실제 경로 overlay | sim-to-real gap 시각화 |
| shot history table | 데이터 기반 보정 근거 |

강조 문장:

> 성공률을 높이기 위해 환경을 바꾸는 대신, 실제 타격 결과와 예측 결과의 차이를 측정하여 다음 shot 조건을 보정한다.

---

### 7.7 트릭샷으로 확장성 제시

최소 시도 포켓팅은 game-level planning을 보여주고, 트릭샷은 다중 물리 상호작용을 보여준다.

강조 문장:

> 메인 데모는 sequential pocketing을 통해 게임 수행 능력을 보여주고, 트릭샷은 한 번의 타격으로 여러 공에 영향을 주는 multi-object physical task로 확장 가능성을 보여준다.

---

## 8. 최종 발표 스토리라인

### Slide 1. Motivation

- 기존 포켓볼 로봇은 주로 공 하나를 포켓에 넣는 데모 중심
- 우리는 기존 포켓볼대 환경을 바꾸지 않고, 로봇이 현재 상태에서 최적의 shot을 선택하도록 한다.

### Slide 2. Problem Definition

입력:

- cue ball 위치
- object ball 위치들
- 6개 포켓 위치
- 로봇 현재 상태

출력:

- 선택된 object ball
- 선택된 target pocket
- ghost ball 위치
- strike direction
- robot trajectory

목표:

- 최소한의 시도로 모든 공을 포켓팅
- 실패 시 재인식 및 재계획

### Slide 3. System Pipeline

```text
Camera Perception
→ Coordinate Calibration
→ Candidate Shot Generation
→ Shot Scoring
→ Robot Feasibility Check
→ Strike Execution
→ Re-detection
→ Adaptive Correction
```

### Slide 4. Greedy Shot Planner

- 목적구 N개와 포켓 6개 조합 생성
- 총 후보 수 = N × 6
- 각 후보에 대해 score 계산
- 최고 score shot 선택

### Slide 5. Robot-aware Execution

- ghost ball 위치 계산
- 인디7 접근 가능성 확인
- ㄴ자형 툴팁으로 수평 타격
- moveL 또는 속도 제어 기반 strike

### Slide 6. Sim-to-real Adaptation

- 예측 결과와 실제 결과 비교
- 포켓팅 실패 방향 또는 target error 측정
- 다음 shot에서 angle/speed/ghost ball offset 보정

### Slide 7. Demo 1: Minimum-shot Pocketing

- 2~3개 목적구 대상
- 모든 공-포켓 후보 평가
- 가장 쉬운 shot부터 선택
- 매 shot 후 재인식 및 재계획

### Slide 8. Demo 2: Trick Shot

- 한 번의 타격으로 두 공 모두 맞히기
- 또는 두 공을 각각 target zone으로 이동
- path planning과 다중 충돌 task 강조

### Slide 9. Expected Contribution

- 기존 환경을 바꾸지 않는 robot adaptation
- 6개 포켓 기반 greedy shot selection
- 인디7 실행 가능성을 고려한 robot-aware planning
- 실패 후 재인식 및 보정하는 closed-loop 구조
- 단순 포켓팅을 넘어 트릭샷으로 확장

---

## 9. 교수님께 말할 최종 답변 스크립트

> 저희는 포켓볼대 표면이나 쿠션을 바꾸지 않고, 기존 환경 그대로에서 로봇이 task를 수행하도록 방향을 수정했습니다. 메인 데모는 여러 목적구와 6개 포켓 조합을 모두 후보로 생성한 뒤, cut angle, 거리, 장애물 여부, 로봇 실행 가능성, 필요한 속도를 기준으로 가장 성공 가능성이 높은 shot을 선택하는 것입니다. 한 번 타격한 뒤에는 공 배치를 다시 인식하고 다음 shot을 재계획하여 최소한의 시도로 모든 공을 포켓에 넣는 것을 목표로 합니다.
>
> 추가로 트릭샷 데모에서는 한 번의 타격으로 두 개의 공에 영향을 주는 task를 보여주려고 합니다. 이를 통해 단순 포켓팅뿐 아니라 다중 물리 상호작용을 고려한 로봇 타격 planning까지 보여줄 수 있습니다. 핵심은 성공률을 높이기 위해 환경을 바꾸는 것이 아니라, 로봇이 현재 환경의 오차를 관찰하고 다음 shot에 반영하는 closed-loop robotic billiards system을 보여주는 것입니다.

---

## 10. 최종 결론

최종 데모의 핵심은 다음 세 가지다.

1. **Minimum-shot pocketing**
   - 여러 공과 6개 포켓 후보 중 가장 성공 가능성이 높은 shot을 선택한다.

2. **Closed-loop re-planning**
   - 한 번 친 뒤 공 배치를 다시 인식하고 다음 shot을 재계획한다.

3. **Trick shot extension**
   - 단순 포켓팅을 넘어 한 번의 타격으로 여러 공에 영향을 주는 task를 보여준다.

최종 한 문장:

> 기존 포켓볼대 환경을 바꾸지 않고, 인디7이 현재 공 배치에서 가장 성공 가능성이 높은 shot을 선택하고, 매 타격 후 재인식과 보정을 통해 최소한의 시도로 모든 공을 포켓팅하는 로봇팔 포켓볼 시스템을 구현한다.
