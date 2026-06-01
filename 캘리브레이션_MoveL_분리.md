# MoveL 캘리브레이션 — Approach / Strike 분리

## 목적

실기 타격 시 **movej 곡선으로 접근 후 movel** 이 아니라,

1. **Approach** = `movel`만 → 로봇 **완전 정지**
2. 사람이 확인 후 **[Enter] = START**
3. **Strike** = `movel` **직선 1번**만
4. **Retract** = `movel` 상승
5. (trial 끝) **Home** = `movej`만

**Approach ↔ Strike 사이에는 `movej` 없음.**

---

## 관련 파일

| 파일 | 역할 |
|------|------|
| `calibration_loop_movel.py` | Phase 1 좌표 / Phase 2 물리 캘리브레이션 |
| `project/real_movel_player.py` | Approach·Strike movel, 계획 저장/로드 |
| `last_calib_shot.npz` | Approach 후 저장되는 Ready/Strike 명령 |
| `calibration_position_offset.json` | Phase 1 좌표 오프셋 |
| `calibration_trials.npz` | Phase 2 실측 trial |
| `calibration_result_physics.npz` | Phase 2 최적화 결과 |

---

## 실행 모드 (`--real-step`)

| 값 | 설명 |
|----|------|
| `full` (기본) | Approach → **Enter** → Strike → Retract (한 프로세스) |
| `approach` | Approach + Align만 → **정지** → `last_calib_shot.npz` 저장 후 종료 |
| `strike` | 저장된 Ready에서 **Enter** → Strike → Retract (별도 명령) |

---

## Phase 1: 좌표 캘리브레이션

### A) 한 세션 (`full`)

```bash
cd code
python calibration_loop_movel.py --phase position --robot-ip 192.168.0.13
```

매 샷: 시뮬 미리보기 → Approach movel → **Enter** → Strike movel → `l/r/g/m` 입력 → Home.

### B) 실행 분리 (권장)

**+x 약타격 (y offset 보정)**

```bash
# 1) 접근만 — 로봇 Ready에서 멈춤
python calibration_loop_movel.py --phase position --real-step approach --axis x

# 2) 확인 후 START → 타격 → 키보드 입력
python calibration_loop_movel.py --phase position --real-step strike
```

**+y 약타격 (x offset 보정)**

```bash
python calibration_loop_movel.py --phase position --real-step approach --axis y
python calibration_loop_movel.py --phase position --real-step strike
```

`g`(직진) 입력 시 해당 축 보정 완료. `l`/`r`/`m`이면 offset 저장 후 다시 approach/strike 반복.

---

## Phase 2: 물리 캘리브레이션

### A) 한 세션

```bash
python calibration_loop_movel.py --phase physics --num-trials 5
```

### B) trial마다 분리

```bash
# Trial 1 — 접근만
python calibration_loop_movel.py --phase physics --real-step approach

# Trial 1 — START 후 타격 + 카메라 관측 + trial 저장
python calibration_loop_movel.py --phase physics --real-step strike

# Trial 2 … approach / strike 반복
```

모든 trial 후 최적화만:

```bash
python calibration_loop_movel.py --optimize-only
```

---

## 화면 메시지 (Strike 직전)

```
========================================================
  APPROACH 완료 — 로봇 정지 (이 구간에 movej 없음)
  큐·큐대·자세 확인 후
  >>> [Enter] = START → MoveL 직선 STRIKE 만 실행
========================================================
```

**Enter 전에는 Strike 명령이 나가지 않습니다.**

---

## movej가 쓰이는 경우

| 시점 | 명령 |
|------|------|
| 프로그램 시작 (FK offset) | `movej` 홈 |
| 각 trial / 샷 끝 | `movej` 홈 |
| Approach ~ Strike 사이 | **사용 안 함** |

---

## 주요 옵션

| 옵션 | 설명 |
|------|------|
| `--robot-ip` | Indy IP (기본 `192.168.0.13`) |
| `--real-step full\|approach\|strike` | 실행 분리 |
| `--plan-file` | 계획 npz 경로 (기본 `last_calib_shot.npz`) |
| `--axis x\|y` | position + approach 시 필수 |
| `--skip-fk-offset` | FK TCP 보정 생략 |
| `--allow-auto-strike` | full 모드에서 Enter 생략 (비권장) |
| `--test` | 로봇 없이 시뮬 loss 확인 |

---

## 변경 요약 (2026-05)

1. **`calibration_loop_movel.py`** 추가 — MoveL 기반 캘리브레이션
2. **`project/real_movel_player.py`** — Approach/Strike 분리, `save_shot_plan` / `load_shot_plan`
3. **`--real-step approach | strike`** — 실행을 두 번으로 나눔
4. Approach↔Strike 사이 **movej 제거**, Enter = START
5. `calibration_loop_fix.py`는 머지 충돌 상태 → **이 스크립트 사용 권장**

---

## 문제 해결

| 증상 | 확인 |
|------|------|
| Strike 안 나감 | Enter 눌렀는지, `strike_dist < 3mm` 로그 (skip) |
| approach 후 홈 감 | `--real-step strike`만 실행했는지 (approach 직후는 Ready 유지) |
| `계획 없음` | 먼저 `--real-step approach` 실행 |
| 로봇 안 움직임 | IP, Teleop/IDLE 모드, `indy.recover()` |

---

## 참고

- 단발 테스트: `real_approach_then_strike.py` (`--phase approach` / `strike`)
- 시뮬·홀 회피 등: `docs/시뮬_실기_변경_정리.md`
