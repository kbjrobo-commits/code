"""
인식(Perception) 인터페이스
============================
시뮬과 실제 모드 전환을 위한 추상 레이어.
- SimPerception: PyBullet API로 읽기
- RealPerception: RealSense + IndyEye (스텁 → 카메라 연결 시 구현)

기획서 2.2절: Coarse-to-Fine 아키텍처
"""
import numpy as np
from abc import ABC, abstractmethod


class PerceptionInterface(ABC):
    """인식 추상 인터페이스"""

    @abstractmethod
    def scan_environment(self):
        """환경 스캔 — 공, 장애물 좌표 반환

        Returns:
            dict: {
                'cue_pos': [x,y,z],
                'target_pos': [x,y,z],
                'obstacles': [(x,y,r), ...],
                'table_bounds': {...}
            }
        """
        pass

    @abstractmethod
    def observe_result(self):
        """타격 후 결과 관찰

        Returns:
            dict: {
                'target_hit': bool,
                'cue_pos': [x,y,z],
                'target_pos': [x,y,z],
                'distance': float
            }
        """
        pass


class SimPerception(PerceptionInterface):
    """시뮬레이션 인식 — PyBullet API로 직접 읽기"""

    def __init__(self, environment):
        """
        Args:
            environment: MazeEnvironment (or BilliardsEnvironment, MiniGolfEnvironment)
        """
        self.env = environment

    def scan_environment(self):
        cue_pos = self.env.get_cue_ball_position()
        target_pos = self.env.get_target_ball_position()
        obstacles = self.env.get_obstacle_positions()
        return {
            'cue_pos': cue_pos,
            'target_pos': target_pos,
            'obstacles': obstacles,
            'table_bounds': getattr(self.env, 'table_bounds', None)
        }

    def observe_result(self):
        self.env.wait_balls_stop(timeout=8.0)
        cue_pos = self.env.get_cue_ball_position()
        target_pos = self.env.get_target_ball_position()
        target_hit = self.env.is_target_hit()
        dist = np.linalg.norm(cue_pos[:2] - target_pos[:2])
        return {
            'target_hit': target_hit,
            'cue_pos': cue_pos,
            'target_pos': target_pos,
            'distance': dist
        }


class RealPerception(PerceptionInterface):
    """실제 카메라 인식 — RealSense(Global) + IndyEye(Local)

    현재: 스텁 구현 (수동 좌표 입력)
    향후: 카메라 연결 시 detect_ball(), detect_obstacles() 구현
    """

    def __init__(self, camera=None, table_bounds=None):
        """
        Args:
            camera: RealSense 인스턴스 (None이면 수동 입력 모드)
            table_bounds: 테이블 경계 dict
        """
        self.camera = camera
        self.table_bounds = table_bounds
        self._manual_obstacles = []
        self._manual_cue = None
        self._manual_target = None

    def set_manual_positions(self, cue_pos, target_pos, obstacles):
        """수동 좌표 입력 (카메라 없을 때)"""
        self._manual_cue = np.array(cue_pos)
        self._manual_target = np.array(target_pos)
        self._manual_obstacles = list(obstacles)

    def scan_environment(self):
        if self.camera is not None:
            return self._scan_with_camera()
        else:
            return self._scan_manual()

    def _scan_manual(self):
        """수동 입력 기반 스캔"""
        if self._manual_cue is None:
            raise RuntimeError("RealPerception: 카메라 없음. set_manual_positions() 호출 필요")
        return {
            'cue_pos': self._manual_cue,
            'target_pos': self._manual_target,
            'obstacles': self._manual_obstacles,
            'table_bounds': self.table_bounds
        }

    def _scan_with_camera(self):
        """카메라 기반 스캔 — 향후 구현

        구현 시 필요한 것:
        1. self.camera.get_color_depth() → RGB + Depth
        2. detect_obstacles(rgb, depth, intrinsic) → [(x,y,r), ...]
        3. detect_ball(rgb, depth, intrinsic, color='white') → [x,y,z]
        4. detect_ball(rgb, depth, intrinsic, color='red') → [x,y,z]
        5. pixel → world 좌표 변환 (intrinsic + extrinsic)
        """
        # TODO: 카메라 연결 시 구현
        # rgb, depth, _ = self.camera.get_color_depth()
        # obstacles = detect_obstacles(rgb, depth, self.camera.camera_matrix)
        # cue_pos = detect_ball(rgb, depth, self.camera.camera_matrix, 'white')
        # target_pos = detect_ball(rgb, depth, self.camera.camera_matrix, 'red')
        raise NotImplementedError("카메라 기반 스캔 미구현 — detect_obstacles/detect_ball 구현 필요")

    def observe_result(self):
        if self.camera is not None:
            return self._observe_with_camera()
        else:
            # 수동 모드: 사용자가 결과를 육안 확인
            print("[RealPerception] 타격 결과를 육안으로 확인하세요.")
            hit = input("  목표공 명중? (y/n): ").strip().lower() == 'y'
            return {
                'target_hit': hit,
                'cue_pos': self._manual_cue,
                'target_pos': self._manual_target,
                'distance': 0.0
            }

    def _observe_with_camera(self):
        """카메라 기반 관찰 — 향후 구현"""
        raise NotImplementedError("카메라 기반 관찰 미구현")
