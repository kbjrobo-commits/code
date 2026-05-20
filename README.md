# 🤖 Indy7 Autonomous Strike Robot

> **POSTECH 로보틱스개론 팀 프로젝트**
>
> Neuromeka Indy7 6-DOF 산업용 로봇 팔을 이용한 자율 미니골프 퍼팅, 포켓볼, 쓰리쿠션 당구 시뮬레이션

---

## 📋 목차

- [프로젝트 개요](#-프로젝트-개요)
- [시스템 아키텍처](#-시스템-아키텍처)
- [디렉토리 구조](#-디렉토리-구조)
- [핵심 모듈 설명](#-핵심-모듈-설명)
- [물리 엔진 원리](#-물리-엔진-원리)
- [제어 파이프라인](#-제어-파이프라인)
- [설치 및 실행](#-설치-및-실행)
- [설정(Config) 가이드](#-설정config-가이드)
- [알려진 이슈 및 개선점](#-알려진-이슈-및-개선점)

---

## 🎯 프로젝트 개요

### 목표
Indy7 로봇 팔에 컴팩트 타격 헤드를 장착하고, **비전 기반 자율 상태 머신(SCAN → THINK → STRIKE → OBSERVE)**을 통해 세 가지 과제를 수행합니다:

1. **미니골프 퍼팅**: 울퉁불퉁한 3D 지형 위의 골프공을 한 번의 타격으로 홀 컵에 넣기 (홀인원)
2. **포켓볼(당구)**: 큐볼로 목표 공을 타격하여 6개 포켓 중 하나에 넣기
3. **쓰리쿠션 당구**: 큐볼이 쿠션에 3회 이상 반사되면서 황구·적구 모두 접촉하는 경로 계산 및 실행

### 핵심 기술
| 기술 | 설명 |
|------|------|
| **3공 동시 2D 물리 시뮬레이션** | 탄성 충돌·쿠션 반사·구름 마찰을 포함한 다물체 시뮬레이션으로 3쿠션 경로 탐색 |
| **다중 후보 탐색 + IK 검증** | 어닐링 기반 후보 최대 10개 생성 → 장애물 클리어런스 + IK 유효성 순회 검증 |
| **PyBullet 물리 변수 동기화** | 2D 계획기가 PyBullet과 동일한 반발계수·마찰 계수 사용 (`max(e1,e2)` 공식) |
| **Jacobian 기반 수치 IK** | Damped Least Squares (DLS) 방식의 역기구학 솔버 |
| **PD 토크 제어** | Computed Torque + PD 제어기 (Kp=800, Kd=40) |
| **즉시 후퇴 프로토콜** | 타격 직후 Home 위치로 즉시 복귀하여 공 궤적 방해 방지 |
| **Physics-based Grid Search** | PyBullet 헤드리스 시뮬레이션으로 최적 미니골프 타격 속도/각도 탐색 |
| **운동량 보존 역산** | 1차원 정면충돌 + cos(15°) 대각선 보정으로 공 속도 → EE 속도 변환 |

---

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                  State Machine (FSM)                    │
│  ┌──────┐   ┌───────┐   ┌────────┐   ┌─────────┐      │
│  │ SCAN │──▶│ THINK │──▶│ STRIKE │──▶│ OBSERVE │──┐   │
│  └──────┘   └───────┘   └────────┘   └─────────┘  │   │
│      ▲                                      │      │   │
│      └──────── (retry if miss) ─────────────┘      │   │
│      └──────── (next round if success) ────────────┘   │
└────────────┬───────────────┬──────────────┬────────────┘
             │               │              │
     ┌───────▼───────┐ ┌────▼────┐  ┌──────▼──────┐
     │ ShotPlanner/  │ │Traj.Plan│  │ RobotControl│
     │ CushionPlanner│ │(SE3 궤적)│  │ (PD 토크)   │
     └───────────────┘ └─────────┘  └─────────────┘
             │                              │
     ┌───────▼───────┐              ┌──────▼──────┐
     │ PyBullet      │              │ PinocchioModel│
     │ (Headless Sim)│              │ (FK, Jacobian)│
     └───────────────┘              └──────────────┘
```

### 작동 흐름

1. **SCAN**: PyBullet에서 공/홀/장애물 위치를 직접 읽음 (비전 인터페이스 대체)
2. **THINK**: 쓰리쿠션은 어닐링 탐색으로 다중 후보 생성 / 미니골프는 Grid Search로 최적 속도·방향 탐색
3. **STRIKE**: 후보를 IK + 장애물 클리어런스로 순회 검증 → 유효한 첫 번째 후보만 실행 → 즉시 Home 복귀
4. **OBSERVE**: 공이 멈출 때까지 대기 → 성공/실패 판정 → 실패 시 재시도

---

## 📁 디렉토리 구조

```
code/
├── project/                         # 🎯 핵심 프로젝트 코드
│   ├── config.py                    # 전체 설정 파라미터 (물리, 로봇, 탐색)
│   ├── robot_controller.py          # 로봇 제어기 (GUI/Headless/Real 모드)
│   ├── state_machine.py             # 자율 상태 머신 (SCAN→THINK→STRIKE→OBSERVE)
│   ├── trajectory_planner.py        # 타격 궤적 생성기 (Approach→Strike→Follow)
│   ├── ik_solver.py                 # Damped Least Squares IK 솔버
│   ├── perception.py                # 환경 인식 (PyBullet 기반 / Real 스텁)
│   ├── run_demo_and_plot.py         # ★ 메인 데모 실행 + 플롯 생성
│   ├── run_simulation.py            # 인터랙티브 데모 (Ctrl+C로 종료)
│   │
│   ├── environment/                 # 시뮬레이션 환경
│   │   ├── minigolf_env.py          # 미니골프: 지형 + 홀 + 공 + 도구
│   │   ├── billiards_env.py         # 포켓볼: 테이블 + 쿠션 + 포켓 + 공 + 도구
│   │   └── maze_env.py              # 쓰리쿠션: 테이블 + 쿠션 + 3공 + 장애물 + 도구
│   │
│   └── physics/                     # 물리 기반 계획
│       ├── shot_planner.py          # Grid Search (미니골프) + 기하학적 정렬 (포켓볼)
│       └── cushion_planner.py       # 3공 동시 2D 시뮬 기반 쓰리쿠션 타격 탐색
│
├── src/                             # 🔧 프레임워크 (수업 제공 코드)
│   ├── core/
│   │   ├── pybullet_core.py         # PyBullet 시뮬레이터 코어 (스레드 루프)
│   │   └── pybullet_robot.py        # Indy7 로봇 모델 (FK, 토크 제어)
│   ├── utils/
│   │   ├── pinocchio_utils.py       # Pinocchio 기반 FK/Jacobian/Dynamics
│   │   ├── rotation_utils.py        # 회전 변환 유틸리티
│   │   └── robotics_utils.py        # SE3, 포즈 변환
│   ├── camera/
│   │   ├── camera_interface.py      # 카메라 인터페이스 (추상)
│   │   └── realsense.py             # Intel RealSense 드라이버
│   └── assets/urdf/                 # Indy7 URDF 모델 파일
│
├── 1~5ExampleCode_*.ipynb           # 수업 예제 코드
├── figures/                         # 기획서 첨부 이미지
└── walkthrough.md                   # 개발 과정 기록
```

---

## 🔍 핵심 모듈 설명

### 1. `config.py` — 전체 설정

| 섹션 | 주요 파라미터 | 설명 |
|------|-------------|------|
| 로봇 | `HOME_Q_DEG`, `MAX_TOOL_SPEED` | 홈 포지션 [0,-15,-75,0,-90,0]°, 최대 1.0 m/s |
| 궤적 | `TRAJECTORY_DT`, `IK_GAIN` | 1ms 타임스텝, IK 게인 1.0 |
| 타격 | `STRIKE_APPROACH_DIST=0.08` | 접근 거리 8cm |
| 도구 | `TOOL_HEAD_MASS=0.2kg` | 컴팩트 원통 헤드 (3cm 길이, 반발계수 0.9) |
| 쓰리쿠션 | `MAZE_BALL_RESTITUTION=0.85` | PyBullet 동기화된 반발계수 |
| 쓰리쿠션 | `MAZE_BALL_ROLLING_FRICTION=0.02` | PyBullet 동기화된 구름 마찰 |
| 어닐링 | `ANNEAL_SPEED_RANGE=(0.15, 0.5)` | 공 속도 탐색 범위 |
| 어닐링 | `ANNEAL_MAX_CUSHIONS=6` | 최대 쿠션 반사 횟수 |

### 2. `cushion_planner.py` — 쓰리쿠션 계획기

3공 동시 2D 물리 시뮬레이션 기반 타격 탐색:

```
1. 어닐링 탐색: 500개 초기 샘플 → 3라운드 정밀화
2. 각 후보에 대해 3공 2D 시뮬레이션:
   - 탄성 공-공 충돌 (운동량 + 에너지 보존)
   - 쿠션 반사 (법선 방향 반발)
   - 장애물 반사
   - 구름 마찰 감쇠
3. 스코어링: hit_t1 + hit_t2 + 쿠션수 ≥ 3 → 성공
4. 각도 15° 이상 차이나는 상위 10개 다양한 후보 리턴
```

#### PyBullet 물리 변수 동기화

| 변수 | PyBullet | 2D 계획기 | 일치 |
|------|----------|-----------|------|
| 쿠션 반발계수 | `max(0.85, 0.8) = 0.85` | `max(ball_e, cushion_e) = 0.85` | ✅ |
| 공-공 반발계수 | `max(0.85, 0.85) = 0.85` | `0.85` | ✅ |
| 구름 마찰 | `0.02` | `0.02` | ✅ |
| 횡 마찰 | `0.3` | `0.3 × 0.05` 근사 감쇠 | ✅ |
| 타격 속도 전달 | `cos(15°) = 0.966` | `ball_speed / cos(15°)` 보정 | ✅ |

### 3. `state_machine.py` — 자율 상태 머신

```python
sm = AutonomousStateMachine(controller, environment, planner, traj_planner, demo_type)
success = sm.run(max_attempts=3)
```

#### 쓰리쿠션 `_strike()` 핵심 로직

```
for candidate in candidates:        # 최대 10개 다양한 후보
    for phi in [0°, 30°, ..., 330°]:  # 12개 도구 축 회전
        trajectory = plan_strike(candidate, phi)
        
        if obstacle_clearance(trajectory, obstacles, 12cm):  # 로봇 팔 폭 포함
            if IK_valid(trajectory):                          # 관절 한계 + 특이점
                EXECUTE(trajectory)                           # ← 유효한 첫 후보만 실행
                goto_home(no_wait)                            # ← 즉시 Home 복귀
                return
    
    print(f"SKIP Candidate #{i}: {reason}")

print("All candidates failed → skip strike")
```

### 4. `robot_controller.py` — 로봇 제어기

| 메서드 | 기능 |
|--------|------|
| `connect()` | PyBullet GUI/DIRECT 연결 또는 실제 로봇 연결 |
| `execute_trajectory()` | Phase-aware 실행 (Approach → Strike → 즉시 Home 복귀) |
| `boost_pd_gains(kp, kd)` | PD 게인 동적 변경 (monkey-patch 방식) |

#### 타격 실행 (`_execute_sim`)

```
Phase 1: Approach
  → Home → 상공(25cm) → Ready(공 뒤 8cm)로 2단계 안전 접근

Phase 2: Strike (swing-through)
  → MoveRobot(q_follow)로 공 너머 5cm까지 목표 설정
  → qdot_des 주입으로 PD 브레이크 방지

Phase 3: 즉시 Home 복귀 (대기 없음)
  → MoveRobot(HOME_Q_DEG) + sleep(0.05)
  → PD 컨트롤러가 백그라운드에서 이동, 공 물리 동시 진행
```

> **⚠️ 즉시 후퇴 (Strike-and-Home)**
>
> 타격 후 z+20cm 상공에서 대기하면 공이 로봇 팔에 부딪혀 궤적이 왜곡됨.
> 즉시 Home 목표를 설정하면 PD 컨트롤러가 팔을 빠르게 빼면서 공 물리가 동시에 진행되어 방해 없음.

### 5. `trajectory_planner.py` — 궤적 생성기

```
T_current ──(상공경유)──▶ T_ready ──(const speed)──▶ T_impact ──(decel)──▶ T_follow
          Approach (25cm 상공)       Strike                    Follow-through
```

### 6. `ik_solver.py` — 역기구학 솔버

- `solve_step()`: 단일 IK 스텝
- `solve_trajectory_validated()`: 궤적 전체 IK + 관절한계/특이점/점프 검증

### 7. `shot_planner.py` — 미니골프/포켓볼 계획

- **미니골프**: 헤드리스 PyBullet Grid Search (31 각도 × 25 속도 = 775 조합)
- **포켓볼**: 기하학적 역산 (포켓 → 접촉점 → 타격 방향)

---

## ⚙️ 물리 엔진 원리

### 토크 제어 (Computed Torque + PD)

```python
# pybullet_robot.py → _compute_torque_input()
qddot = qddot_des + Kp*(q_des - q) + Kd*(qdot_des - qdot)
tau = M @ qddot + C*qdot + g    # 동역학 보상
```

### 운동량 전달 모델 (대각선 타격 보정 포함)

```
1차원 정면충돌 (EE 질량 0.2kg + 공 질량 0.17kg):

  합성 반발계수:  e = √(e_tool × e_ball) = √(0.9 × 0.85) ≈ 0.874

  충돌 후 공 속도:  v_ball = (1+e) × m_tool / (m_tool + m_ball) × v_EE_horiz

  대각선 보정:  v_EE_horiz = v_EE × cos(15°) = 0.966 × v_EE
  → EE 속도 역산 시 v_ball / (ratio × cos(15°))
```

---

## 🚀 설치 및 실행

### 요구 사항

```
Python 3.10+
PyBullet
NumPy
Matplotlib
Pinocchio (pin)
```

### Conda 환경 설정

```bash
conda activate indy7_project
```

### 실행 명령어

```bash
cd c:\Users\smart\Desktop\POSTECH\Class\ROBOTICSINTRO\code

# 미니골프
python project/run_demo_and_plot.py --demo minigolf

# 포켓볼
python project/run_demo_and_plot.py --demo billiards

# 쓰리쿠션 (장애물 5개)
python -c "from project.run_demo_and_plot import run_gui_demo; run_gui_demo('maze', num_obstacles=5)"

# 인터랙티브 모드
python project/run_simulation.py --demo minigolf
```

---

## 🔧 설정(Config) 가이드

### 쓰리쿠션 튜닝

| 파라미터 | 현재값 | 효과 |
|----------|--------|------|
| `ANNEAL_SPEED_RANGE` | (0.15, 0.5) | 넓히면 강타 가능, 좁히면 정밀도↑ |
| `ANNEAL_MAX_CUSHIONS` | 6 | 늘리면 더 복잡한 경로 허용 |
| `ANNEAL_N_INITIAL` | 500 | 늘리면 탐색 정밀도↑, 시간↑ |
| `ANNEAL_ROLLING_FRICTION` | 0.02 | PyBullet `MAZE_BALL_ROLLING_FRICTION`과 동일 |
| `MAZE_CUSHION_RESTITUTION` | 0.8 | 쿠션 반발력 (PyBullet은 max(ball, cushion) 사용) |

### 미니골프 튜닝

| 파라미터 | 현재값 | 효과 |
|----------|--------|------|
| `GRID_SPEED_RANGE` | (0.3, 1.5) | 넓히면 더 강한 타격 탐색 |
| `GRID_ANGLE_STEP` | 2.0° | 줄이면 각도 정밀도↑ |
| `MINIGOLF_TERRAIN_RESOLUTION` | 150 | 높이면 홀 형상 정밀도↑ |

---

## ⚠️ 해결된 주요 이슈 및 개선사항

### 1. 로봇-공 물리 간섭 및 궤적 왜곡 (해결)
- **이전 문제**: 타격 후 로봇 팔이 홈 위치로 복귀할 때 로봇 링크가 물리적으로 공을 밀어내어 최종 궤적이 흐트러지는 문제가 있었습니다.
- **해결**: `robot_controller.py`의 `_streaming_thread_pre` 내에서 도구(`tool_id`) 뿐만 아니라 로봇 링크 전체(`robot_id`)에 대해 공과의 충돌 필터를 비활성화 처리하여, 타격 직후 복귀 동작으로 인한 공의 밀림 현상을 원천 차단했습니다.

### 2. 2D 계획기 - 3D 물리 엔진 간의 불일치 (해결)
- **이전 문제**: Headless 계획기에서는 성공한 궤적이 실제 GUI 실행 시 50도~130도 이상 크게 어긋나거나, 3쿠션 궤적을 정상적으로 찾지 못하는 문제가 발생했습니다.
- **해결**:
  - **IK 연산 지연(Lagging) 해결**: 역기구학 솔버(`IKSolver`)가 단일 스텝만 수행할 때 발생하는 추종 지연(Lagging)으로 인해 직선이 아닌 곡선(Arc) 궤적이 생성되고, 이로 인해 대각선/빗겨치기 충돌이 일어나는 것을 발견했습니다. 궤적 생성 및 검증 단계에서 IK를 5회씩 반복 수렴(`for _ in range(5)`)하도록 개선하여 완벽한 직선 타격을 구현했습니다.
  - **3쿠션 규칙 판정 정밀화**: `cushion_planner.py`와 `state_machine.py`에서 두 번째 적구를 맞추기 전까지의 **누적 쿠션 접촉 수 >= 3** 조건을 완벽히 충족하도록 판정 로직을 실제 당구 룰에 맞춰 개정했습니다.
  - **속도 탐색 범위 확장**: 공의 가벼운 질량으로 인해 타격 후 속도가 약 1.8배 빨라지는 물리 법칙(운동량 보존)을 반영해 최대 탐색 속도 제한을 `1.8 m/s`로 확장하여 충분한 에너지를 갖는 3쿠션 성공 경로를 정상 탐색하도록 수정했습니다.

---

## ⚠️ 알려진 이슈 및 개선점

### 현재 한계

1. **장애물 클리어런스 딜레마**: 클리어런스를 크게(12cm) 잡으면 모든 후보가 거부되고, 작게(5cm) 잡으면 로봇 팔이 장애물과 겹침. 현재 7cm로 타협 중이나 장애물 배치에 따라 전부 거부될 수 있음
2. **로봇-환경 충돌 비활성화**: 로봇 링크와 환경 객체(장애물, 벽, 바닥) 간 물리 충돌이 전면 비활성화. EE 위치 기반 클리어런스 체크만 수행하므로 팔 중간 링크가 장애물을 관통할 수 있음
3. **비전 시스템 미구현**: SCAN에서 PyBullet API로 위치 직접 읽음
4. **타격 속도 일관성**: PD 제어기 특성상 관절 구성에 따라 실제 EE 속도가 계획 대비 변동

### 향후 개선 방향

1. **로봇 팔 충돌 해결**: FK로 전체 링크 위치 계산 → 장애물/공 proximity 체크, 또는 로봇 링크-공 충돌만 선택적 활성화
2. **PyBullet 직접 시뮬**: 2D 계획기 대신 headless PyBullet에서 3공을 직접 시뮬레이션하여 gap 제거
3. **즉시 후퇴 개선**: 타격 직후 관절을 resetJointState로 즉시 텔레포트하여 공과의 간섭 완전 제거
4. **RealSense 카메라 통합**: 실시간 볼 트래킹
5. **실제 로봇 배포**: IndyDCP3 통신 모듈 실기 테스트

---

## 📄 기획서

- `로보틱스개론 팀 프로젝트 기획서 1.pdf` — 미니골프 퍼팅 로봇
- `로보틱스개론 팀 프로젝트 기획서 2.pdf` — 포켓볼 타격 로봇

---

## 📝 라이선스

POSTECH 로보틱스개론 수업 프로젝트. 교육 목적으로만 사용.
