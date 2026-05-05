# 🤖 Indy7 Autonomous Strike Robot

> **POSTECH 로보틱스개론 팀 프로젝트**
>
> Neuromeka Indy7 6-DOF 산업용 로봇 팔을 이용한 자율 미니골프 퍼팅 및 포켓볼(당구) 타격 시뮬레이션

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
- [기획서](#-기획서)

---

## 🎯 프로젝트 개요

### 목표
Indy7 로봇 팔에 컴팩트 타격 헤드를 장착하고, **비전 기반 자율 상태 머신(SCAN → THINK → STRIKE → OBSERVE)**을 통해 다음 두 가지 과제를 수행합니다:

1. **미니골프 퍼팅**: 울퉁불퉁한 3D 지형 위의 골프공을 한 번의 타격으로 홀 컵에 넣기 (홀인원)
2. **포켓볼(당구)**: 큐볼로 목표 공을 타격하여 6개 포켓 중 하나에 넣기

### 핵심 기술
| 기술 | 설명 |
|------|------|
| **Jacobian 기반 수치 IK** | Damped Least Squares (DLS) 방식의 역기구학 솔버 |
| **PD 토크 제어** | Computed Torque + PD 제어기 (Kp=800, Kd=40) |
| **Physics-based Grid Search** | PyBullet 헤드리스 시뮬레이션으로 최적 타격 속도/각도 탐색 |
| **운동량 보존 역산** | 1차원 정면충돌 모델로 공 속도 → EE 속도 변환 |
| **Concave Trimesh 지형** | 울퉁불퉁한 골프 코스를 오목 메쉬로 생성 (물리적 홀 포함) |

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
     │ ShotPlanner   │ │Traj.Plan│  │ RobotControl│
     │ (Grid Search) │ │(SE3 궤적)│  │ (PD 토크)   │
     └───────────────┘ └─────────┘  └─────────────┘
             │                              │
     ┌───────▼───────┐              ┌──────▼──────┐
     │ PyBullet      │              │ PinocchioModel│
     │ (Headless Sim)│              │ (FK, Jacobian)│
     └───────────────┘              └──────────────┘
```

### 작동 흐름

1. **SCAN**: PyBullet에서 공/홀/포켓 위치를 직접 읽음 (비전 인터페이스 대체)
2. **THINK**: Grid Search로 최적 타격 속도/방향 계산 → 운동량 보존으로 EE 속도 역산
3. **STRIKE**: SE3 궤적 생성 (Approach → Strike → Follow-through) → IK → PD 토크 제어
4. **OBSERVE**: 공이 멈출 때까지 대기 → 홀인원/포켓 판정 → 성공 시 다음 라운드, 실패 시 재시도

---

## 📁 디렉토리 구조

```
code/
├── project/                         # 🎯 핵심 프로젝트 코드
│   ├── config.py                    # 전체 설정 파라미터 (물리, 로봇, Grid Search)
│   ├── robot_controller.py          # 로봇 제어기 (GUI/Headless/Real 모드)
│   ├── state_machine.py             # 자율 상태 머신 (SCAN→THINK→STRIKE→OBSERVE)
│   ├── trajectory_planner.py        # 타격 궤적 생성기 (Approach→Strike→Follow)
│   ├── ik_solver.py                 # Damped Least Squares IK 솔버
│   ├── run_demo_and_plot.py         # ★ 메인 데모 실행 + 플롯 생성
│   ├── run_simulation.py            # 인터랙티브 데모 (Ctrl+C로 종료)
│   │
│   ├── environment/                 # 시뮬레이션 환경
│   │   ├── minigolf_env.py          # 미니골프: 지형 + 홀 + 공 + 도구
│   │   └── billiards_env.py         # 포켓볼: 테이블 + 쿠션 + 포켓 + 공 + 도구
│   │
│   └── physics/                     # 물리 기반 계획
│       └── shot_planner.py          # Grid Search (미니골프) + 기하학적 정렬 (당구)
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
├── 1~5ExampleCode_*.ipynb           # 수업 예제 코드 (Framework, IK, TP, IndyDCP, RealSense)
├── figures/                         # 기획서 첨부 이미지
└── walkthrough.md                   # 개발 과정 기록
```

---

## 🔍 핵심 모듈 설명

### 1. `config.py` — 전체 설정

모든 물리 파라미터, 로봇 파라미터, Grid Search 범위가 한 파일에 집중되어 있습니다.

| 섹션 | 주요 파라미터 | 설명 |
|------|-------------|------|
| 로봇 | `HOME_Q_DEG`, `MAX_TOOL_SPEED` | 홈 포지션 [0,-15,-75,0,-90,0]°, 최대 1.0 m/s |
| 궤적 | `TRAJECTORY_DT`, `IK_GAIN` | 1ms 타임스텝, IK 게인 1.0 |
| 타격 | `STRIKE_APPROACH_DIST` | 접근 거리 0.08m |
| 미니골프 | `MINIGOLF_TERRAIN_RESOLUTION=150` | 150×150 메쉬 (1cm 해상도) |
| 당구 | `BILLIARD_STRIKE_ANGLE_DEG=15` | 15° 대각선 타격 (수평 에너지 96% 보존) |
| 도구 | `TOOL_HEAD_MASS=0.2kg` | 컴팩트 원통 헤드 (3cm 길이, 반발계수 0.9) |
| Grid Search | `GRID_SPEED_RANGE=(0.3, 1.5)` | 탐색 속도 범위 (굴곡 지형 대응) |

### 2. `robot_controller.py` — 로봇 제어기

세 가지 모드를 지원하는 통합 제어기:

```python
controller = RobotController(mode='sim', headless=False)  # GUI 시뮬레이션
controller = RobotController(mode='sim', headless=True)   # 헤드리스 (Grid Search용)
controller = RobotController(mode='real')                  # 실제 로봇 (IndyDCP3)
```

#### 핵심 메서드

| 메서드 | 기능 |
|--------|------|
| `connect()` | PyBullet GUI/DIRECT 연결 또는 실제 로봇 연결 |
| `execute_trajectory()` | SE3 궤적을 phase-aware로 실행 (Approach→Strike→Retract) |
| `boost_pd_gains(kp, kd)` | PD 게인 동적 변경 (monkey-patch 방식) |

#### 타격 실행 원리 (`_execute_sim`)

```
Phase 1: Approach (cubic time scaling)
  → 현재 위치 → 공 뒤 6cm (정밀 접근, 각 웨이포인트에서 time.sleep)

Phase 2: Strike (swing-through)
  → MoveRobot(q_follow) 한 방으로 공 너머 5cm까지 목표 설정
  → qdot_des 주입으로 PD 브레이크 방지
  → swing_time 대기 후 즉시 후퇴

Phase 3: Retract
  → q_ready로 복귀 (준비 위치)
```

> **⚠️ 중요 — PD 브레이크 현상**
>
> PyBullet의 PD 제어기(`_compute_torque_input`)는 `qdot_des = 0`이 기본값입니다.
> 타격 시 EE가 빠르게 움직이면 `Kd * (0 - qdot)` 항이 거대한 역방향 토크를 생성하여
> 로봇이 스스로 브레이크를 밟습니다. 이를 방지하기 위해 **타격 직전에 `_qdot_des`를
> 목표 관절 속도로 주입**하고, 후퇴 시 다시 0으로 복구합니다.

### 3. `state_machine.py` — 자율 상태 머신

```python
sm = AutonomousStateMachine(controller, environment, shot_planner, traj_planner, demo_type)
success = sm.run(max_attempts=3)  # 항상 3라운드 실행
```

- **minigolf**: 홀인원 판정 (`is_hole_in()` — 2D 거리 < 홀 반지름)
- **billiards**: 포켓 판정 (`is_pocketed()` — 목표공이 포켓 반지름 내)
- 큐볼이 도달 범위(0.70m) 밖이면 시작 위치로 자동 리셋

---

## 🎯 최적 타격 속도벡터와 EE 경로를 찾는 전체 흐름

> 이 부분이 이 프로젝트의 핵심입니다. "어떤 방향으로, 얼마나 빠르게 쳐야 하나?"를 찾고,
> 그것을 로봇 팔의 3D 이동 경로로 변환하는 과정 전체를 설명합니다.

### Step 1 — 최적 타격 방향 + 속도 탐색 (`shot_planner.py`)

#### ❓ 핵심 질문: 울퉁불퉁한 지형에서도, 곡선 경로도 어떻게 계산하나요?

**결론: 곡선 경로를 수식으로 계산하지 않습니다. PyBullet이 직접 굴려봅니다.**

울퉁불퉁한 지형 위에서 공이 어떤 경로를 따라가는지를 수학적으로 풀려면
(지형 법선 방향 반력 + 마찰 + 중력 + 구름 운동)을 모두 연립해야 하는데,
이것은 해석적으로 매우 어렵습니다. 대신 이 프로젝트는 다른 접근을 취합니다:

```
┌─────────────────────────────────────────────────────────────────┐
│  "여러 방향과 속도로 공을 수백 번 굴려보고, 가장 결과가 좋은 것을 고른다"  │
└─────────────────────────────────────────────────────────────────┘

물리 엔진(PyBullet)이 알아서 처리하는 것들:
  ✓ 공이 언덕을 넘을 때 속도 감소 (위치에너지 ↑)
  ✓ 내리막에서 가속 (위치에너지 → 운동에너지)
  ✓ 지형 법선 방향의 수직 항력
  ✓ 마찰에 의한 감속
  ✓ 구멍(오목 부분)에 빠져서 멈춤

코드가 하는 일:
  ✓ 초기 속도 벡터를 지정 (방향 + 크기)
  ✓ 2000 스텝 후 공의 위치를 읽어서 홀까지 거리 계산
  ✓ 가장 거리가 짧은 조합을 선택
```

이 방식의 장점은 **지형이 아무리 복잡해도 동일하게 작동**한다는 점입니다.
지형을 변경하면 Grid Search를 다시 실행하기만 하면 되고, 코드 수정이 필요 없습니다.

#### 미니골프: Physics-based Grid Search

별도의 **헤드리스 PyBullet**에서 공을 직접 수백 번 굴려봅니다.

```
[실제 지형 OBJ 메쉬를 headless PyBullet에 로드]  ← GUI와 동일한 지형
                                                   (150×150 해상도, 구멍 포함)
for angle_offset in [-30°, -28°, ..., +28°, +30°]:         ← 31가지 각도
    angle = base_angle(공→홀 방향) + angle_offset
    direction = [cos(angle), sin(angle), 0]

    for speed in [0.30, 0.35, ..., 1.45, 1.50] m/s:         ← 25가지 속도
        
        공 초기 위치 리셋
        공 초기 속도 = direction × speed  ← 이 한 줄만 지정
        
        for step in range(2000):           ← 물리 시뮬레이션 8.3초 분량
            p.stepSimulation()             ← PyBullet이 지형 반력, 마찰, 중력 계산
            if 공이 구멍(Z < -1cm):       ← 홀인원 조기 종료
                공 속도 = 0 (트랩)
                break
        
        final_dist = |공 최종 위치 − 홀 중심|  ← 결과만 읽음

        if final_dist < best_dist:
            best_dir = direction
            best_speed = speed

→ 결과: best_dir (단위 벡터), best_speed (m/s) — 총 775가지 조합 탐색
```

> **왜 초기 위치를 매번 리셋하나요?**
> 각 조합은 독립적으로 실험해야 하므로, 이전 시뮬레이션 결과를 지우고 공을 원래 위치로 돌려놓습니다.
> GUI에서 보이는 지형과 동일한 OBJ 파일을 사용하므로 결과가 실제 시뮬레이션과 일치합니다.

#### ❓ 핵심 질문: 그럼 엄청 느린 거 아닌가요? 시뮬레이션을 그대로 텔레오퍼레이션 하는 건가요?

**결론: 1. 느리지만 (약 3초 소요) 골프나 당구 환경에서는 충분히 수용 가능하며, 2. 텔레오퍼레이션이 아닙니다.**

**1. 속도 문제:**
Grid Search 775번 시뮬레이션은 실제로 **약 3초** 정도 소요됩니다. 
GUI 렌더링이 없는 헤드리스 모드(`DIRECT`)를 사용하기 때문에 실제 시간보다 훨씬 빠르게 연산됩니다. 골프나 당구에서 샷을 준비하는 3초는 허용 가능한 시간이지만, 실시간 회피 반응이 필요한 작업에는 부적합할 수 있습니다.

**2. 텔레오퍼레이션(Teleoperation)과의 차이:**
이 방식은 사람이 원격 조종하는 것이 아니며, **자율 계획(Autonomous Planning) + 자율 실행(Autonomous Execution)**입니다. 이 프로젝트의 접근법은 로보틱스에서 **샘플링 기반 계획(Sampling-based Planning)** 또는 **슈팅 방법(Shooting Method)**과 유사합니다.

| | 텔레오퍼레이션 (원격 조종) | 이 프로젝트 (자율 제어) |
|---|---|---|
| **사람 개입** | 매 순간 사람이 직접 조작 | 없음 (완전 자율) |
| **시뮬레이션 역할** | (보통 사용 안 함) | **타격 파라미터(공의 속도/방향) 탐색에만 사용** |
| **로봇 팔 경로** | 사람의 움직임을 실시간 추적 | **IK + PD 제어로 자율 생성 및 계산** |
| **시뮬 결과 재생?** | N/A | **아니오** — 시뮬레이션 궤적을 그대로 로봇에 복사/재생하는 것이 아닙니다. 시뮬레이션은 공에 줄 "최적 초기 속도"를 찾는 목적일 뿐입니다. |

요약하자면, Grid Search는 머릿속으로 시뮬레이션하며 "어느 방향으로, 얼마나 세게 칠지" 고민하는(THINK) 단계입니다. 결정된 속도와 방향을 바탕으로 실제로 팔을 뻗어 공을 치는(STRIKE) 동작은 별도의 기구학과 동역학 제어기로 계산됩니다.

#### 당구: 기하학적 역산

물리 시뮬레이션 없이 순수 기하학으로 계산합니다.

```
1. 가장 유리한 포켓 선택 (각도 + 거리 최소 점수)
2. 목표공 → 포켓 방향 역산
3. 접촉점 = 목표공 중심 - 2×공반지름 (타격 반대쪽)
4. best_dir = (큐볼 → 접촉점) 방향 단위벡터
```

---

### Step 2 — 공 속도 → EE(엔드이펙터) 속도 역산 (`shot_planner.py`)

Grid Search가 찾은 것은 **공의 속도**입니다. 로봇은 공을 직접 제어할 수 없으므로,
도구(헤드)가 몇 m/s로 움직여야 공이 그 속도로 나가는지를 역산합니다.

```
1차원 정면충돌 운동량 보존:

  합성 반발계수:  e = √(e_tool × e_ball) = √(0.9 × 0.5) ≈ 0.671

  충돌 후 공 속도:  v_ball = (1+e) × m_tool / (m_tool + m_ball) × v_EE
                           = 1.671 × 0.2 / 0.246 × v_EE ≈ 1.358 × v_EE

  역산:  v_EE = v_ball / 1.358

예시: best_speed = 1.3 m/s (공 속도)
  → v_EE = 1.3 / 1.358 ≈ 0.957 m/s (EE가 달성해야 할 임팩트 속도)
```

---

### Step 3 — 2D 방향벡터 → 3D EE 타격 자세 변환 (`trajectory_planner.py`)

Grid Search 결과는 2D 수평 벡터입니다. 이를 3D SE3(위치 + 방향)로 변환합니다.

```python
# 미니골프: 수평 타격
strike_dir_3d = [best_dir[0], best_dir[1], 0.0]

# 당구: 15° 대각선으로 위에서 내려치기
strike_dir_3d = [best_dir[0] × cos(15°),
                 best_dir[1] × cos(15°),
                 -sin(15°)]             ← Z축 음의 방향 (아래로)

# EE의 회전 자세: Z축이 타격 방향을 향하도록
z_axis = strike_dir_3d
x_axis = (world_up × z_axis) / |...|   ← EE 수평 축
y_axis = z_axis × x_axis               ← 오른손 좌표계 완성
R = [x_axis | y_axis | z_axis]         ← 3×3 회전 행렬
```

---

### Step 4 — 3D EE 경로 (SE3 궤적) 생성 (`trajectory_planner.py`)

EE가 공에 도달하기까지의 **3단계 직선 경로**를 생성합니다.

```
현재 EE 위치
    │
    │  [Phase 1: Approach]  (cubic time scaling, 3초)
    │  부드러운 가속→감속으로 준비 위치까지 이동
    ▼
T_ready = 공 위치 − strike_dir × (approach_dist + tool_offset)
          = 공 6cm 뒤 + 도구 길이만큼 후퇴
    │
    │  [Phase 2: Strike]  (일정 속도 v_EE)
    │  v_EE = 0.957 m/s 로 직선 이동
    ▼
T_impact = 공 위치 − strike_dir × tool_offset
           = 도구 끝이 공 표면에 접촉하는 지점
    │
    │  [Phase 3: Follow-through]  (감속)
    │  관통 후 5cm 더 전진 (임팩트 직후 급감속 방지)
    ▼
T_follow = T_impact + strike_dir × follow_dist

총 포인트 수 ≈ 1,600~1,900개 (1ms 간격, 3.3초 궤적)
```

---

### Step 5 — SE3 궤적 → 관절각 변환 (IK) + 토크 제어

```
for T_goal in trajectory:                   ← SE3 목표 포즈 (4×4)
    q_next = IK.solve_step(q_current, T_goal)
    ├── J = A @ Jb   (변환된 Jacobian 6×6)
    ├── error = [위치 오차 3D; 자세 오차 3D]
    └── q += gain × J^T × (J×J^T + λ²I)^{-1} × error    ← DLS IK
    
    robot.MoveRobot(q_next)                 ← 목표 관절각 설정
    ├── q_des ← q_next                      ← PD 제어기 목표 갱신
    └── tau = M×(qddot_des + Kp×Δq + Kd×Δqdot) + C×qdot + g   ← 토크 계산
```

---

### 요약: 찾은 것 → 로봇이 하는 것

```
Grid Search                   IK + PD 제어
─────────────                 ────────────────────────
best_dir (2D 방향)  →  3D SE3 궤적 1,600점  →  관절각 q 1,600점  →  토크 τ 인가
best_speed (공 속도) →  v_EE 역산 0.957m/s  →  Strike 구간 속도 강제
```

---

### 4. `trajectory_planner.py` — 궤적 생성기

3단계 SE3 궤적을 생성합니다:

```
T_current ──(cubic)──▶ T_ready ──(const speed)──▶ T_impact ──(decel)──▶ T_follow
          Approach              Strike                     Follow-through
```

- **Approach**: Cubic time scaling (출발/도착 속도 0) → 부드러운 접근
- **Strike**: 일정 속도 직선 → 공에 정확히 도달
- **Follow-through**: 감속 직선 → 공 너머로 관통

#### `compute_strike_orientation()`
EE의 z축(도구 축)이 타격 방향을 향하도록 회전 행렬 계산. 수평 타격(미니골프)과 대각선 타격(당구) 모두 지원.

### 5. `ik_solver.py` — 역기구학 솔버

Damped Least Squares 방식:

```
q_new = q + gain * J^T * (J*J^T + λ²I)^{-1} * error
```

- `solve_step()`: 단일 IK 스텝 (궤적 추적용)
- `solve_to_target()`: 목표 SE3까지 반복 풀이
- `solve_trajectory()`: 궤적 전체에 대한 IK

### 6. `shot_planner.py` — 물리 기반 타격 계획

#### 미니골프: Grid Search

별도의 **헤드리스 PyBullet 시뮬레이션**에서 공을 직접 굴려봅니다:

1. 지형 OBJ 메쉬를 로드 (concave trimesh)
2. 각도 범위 [-30°, +30°] × 속도 범위 [0.3, 1.5] m/s를 2° × 0.05 간격으로 탐색
3. 각 조합에 대해 공에 초기 속도를 부여하고 2000 스텝 시뮬레이션
4. 홀까지의 최종 거리가 최소인 조합을 선택
5. **공 속도 → EE 속도 역산** (운동량 보존):

```
e = √(e_tool × e_ball)        ← 합성 반발계수
v_ball = (1+e) × m_tool / (m_tool + m_ball) × v_EE
∴ v_EE = v_ball × (m_tool + m_ball) / ((1+e) × m_tool)
```

#### 당구: 기하학적 정렬

1. 6개 포켓 중 가장 유리한 포켓 선택 (각도 + 거리 가중치)
2. 목표공 → 포켓 방향의 반대쪽 접촉점 계산
3. 큐볼 → 접촉점 방향이 타격 방향

### 7. `minigolf_env.py` — 미니골프 환경

#### 지형 생성

```python
_generate_terrain_heightfield(size_x, size_y, resolution, seed, hole_pos, terrain_offset)
```

- Gaussian bump 여러 개를 합성하여 자연스러운 굴곡 생성
- **홀 위치에서 반지름 내 정점의 Z를 -0.05m로 함몰** → 물리적 구멍
- OBJ 파일로 저장 후 `GEOM_FORCE_CONCAVE_TRIMESH`로 로드

#### 홀 트랩 (Trap) 로직

```python
# wait_ball_stop() 내부
if pos[2] < -0.01:  # 공이 지형 아래(구멍 안)로 떨어지면
    resetBaseVelocity(ball, [0,0,0])  # 속도 강제 초기화 → 안착
```

PyBullet의 깔때기 현상(공이 경사를 타고 다시 올라옴)을 방지합니다.

### 8. `billiards_env.py` — 당구 환경

- 테이블: 0.6m × 0.4m 박스 (높이 0.3m)
- 쿠션: 4면 두꺼운 벽 (반발계수 0.8)
- 포켓: 6개 (4코너 + 2사이드), 시각 마커만 (물리적 구멍 없음)
- CCD (Continuous Collision Detection) 활성화 → 고속 공 관통 방지

---

## ⚙️ 물리 엔진 원리

### 토크 제어 (Computed Torque + PD)

```python
# pybullet_robot.py → _compute_torque_input()
qddot = qddot_des + Kp*(q_des - q) + Kd*(qdot_des - qdot)
tau = M @ qddot + C*qdot + g    # 동역학 보상
```

- `M`: 질량 행렬, `C`: 코리올리, `g`: 중력 — Pinocchio로 계산
- `boost_pd_gains()`: Kp=500→800, Kd=20→40으로 동적 강화

### 시뮬레이션 루프

```
PybulletCore._thread_main() (240Hz 별도 스레드)
  └── robot_update()
       ├── _get_robot_states()       ← PyBullet에서 q, qdot 읽기
       ├── _compute_torque_input()   ← PD 토크 계산
       └── _control_robot()          ← TORQUE_CONTROL로 적용
  └── stepSimulation()              ← 물리 엔진 1스텝 전진
```

### 운동량 전달 모델

1차원 정면충돌 (EE 질량 0.2kg + 공 질량 0.046kg):

```
합성 반발계수: e = √(0.9 × 0.5) ≈ 0.671
전달 비율: ratio = (1+e) × m_tool / (m_tool + m_ball)
                 = 1.671 × 0.2 / 0.246 ≈ 1.358
EE 0.957 m/s → 공 1.300 m/s (실제 충돌 결과는 마찰/회전 등으로 약간 감소)
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

# 미니골프만 실행
python project/run_demo_and_plot.py --demo minigolf

# 당구만 실행
python project/run_demo_and_plot.py --demo billiards

# 둘 다 실행
python project/run_demo_and_plot.py --demo both

# 플롯만 생성 (GUI 없이)
python project/run_demo_and_plot.py --plot-only

# 인터랙티브 모드 (Ctrl+C로 종료)
python project/run_simulation.py --demo minigolf
```

### 출력물

| 파일 | 내용 |
|------|------|
| `plot_joints_minigolf.png` | 미니골프 관절각 궤적 (6축) |
| `plot_taskspace_minigolf.png` | 미니골프 태스크 공간 XYZ |
| `plot_3d_minigolf.png` | 미니골프 3D EE 경로 |
| `plot_joints_billiards.png` | 당구 관절각 궤적 |
| `plot_taskspace_billiards.png` | 당구 태스크 공간 XYZ |
| `plot_3d_billiards.png` | 당구 3D EE 경로 |

---

## 🔧 설정(Config) 가이드

### 미니골프 튜닝

| 파라미터 | 현재값 | 효과 |
|----------|--------|------|
| `GRID_SPEED_RANGE` | (0.3, 1.5) | 넓히면 더 강한 타격 탐색, 좁히면 정밀도↑ |
| `GRID_ANGLE_STEP` | 2.0° | 줄이면 각도 정밀도↑, Grid Search 시간↑ |
| `GRID_SIM_STEPS` | 2000 | 늘리면 공이 더 멀리 굴러간 결과까지 고려 |
| `MINIGOLF_TERRAIN_RESOLUTION` | 150 | 높이면 홀 형상 정밀도↑, 메쉬 생성 시간↑ |
| `MINIGOLF_GROUND_FRICTION` | 0.4 | 높이면 공이 빨리 멈춤 (속도↓) |

### 당구 튜닝

| 파라미터 | 현재값 | 효과 |
|----------|--------|------|
| `BILLIARD_STRIKE_SPEED` | 0.8 m/s | 최대 타격 속도 |
| `BILLIARD_STRIKE_ANGLE_DEG` | 15° | 대각선 각도 (수평 에너지 96% 보존) |
| `BILLIARD_BALL_RESTITUTION` | 0.85 | 높이면 충돌 후 속도 유지↑ |

---

## ⚠️ 알려진 이슈 및 개선점

### 현재 한계

1. **비전 시스템 미구현**
   - 현재 SCAN 단계에서 PyBullet API로 공/홀 위치를 직접 읽음
   - 실제 로봇 적용 시 RealSense 카메라 + 컬러 기반 세그멘테이션 필요
   - `src/camera/realsense.py`에 기본 드라이버가 있으나 통합 미완료

2. **타격 속도 일관성**
   - PD 제어기 + `qdot_des` 주입 방식은 관절 구성에 따라 실제 EE 속도가 달라짐
   - Attempt 1에서 계획 대비 ~50% 효율, 이후 시도에서는 잔여 관성으로 과타 가능
   - **개선안**: 속도 기반 제어기(Velocity Control) 또는 토크 FFW(Feed-Forward) 적용

3. **당구 포켓의 물리적 구멍 부재**
   - 현재 포켓은 시각 마커만 있고, 공이 물리적으로 빠지지 않음
   - 미니골프처럼 테이블 메쉬에 물리적 구멍을 뚫거나, 포켓 근처에서 공을 제거하는 로직 필요

4. **Grid Search 계산 시간**
   - 해상도 150 지형 + 775개 조합 탐색 → ~3초 소요
   - 실시간 적용 시 병렬화 또는 학습 기반 정책으로 대체 검토

5. **로봇 도달 범위 제한**
   - Indy7의 유효 도달 범위 약 0.7m
   - 큐볼이 범위 밖으로 가면 리셋 처리 (실제 환경에서는 로봇 이동 필요)

### 향후 개선 방향

1. **RealSense 카메라 통합**: `camera_interface.py` 기반으로 실시간 볼 트래킹
2. **학습 기반 타격 정책**: Grid Search 대신 강화학습(RL) 또는 모방학습 적용
3. **다중 공 당구**: 현재 2공(큐볼+목표공) → 15공 풀게임 확장
4. **실제 로봇 배포**: IndyDCP3 통신 모듈은 구현되어 있으나 실기 테스트 미완료
5. **충돌 회피**: 로봇-환경 충돌을 비활성화하는 대신 정밀 경로 계획으로 회피

---

## 📄 기획서

- `로보틱스개론 팀 프로젝트 기획서 1.pdf` — 미니골프 퍼팅 로봇
- `로보틱스개론 팀 프로젝트 기획서 2.pdf` — 포켓볼 타격 로봇

---

## 📝 라이선스

POSTECH 로보틱스개론 수업 프로젝트. 교육 목적으로만 사용.
