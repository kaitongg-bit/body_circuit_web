"""
身体回路乐队 (Body Circuit Band)
使用姿态识别和人体互动控制音乐播放的互动装置
"""

import cv2
import mediapipe as mp
import numpy as np
import pygame
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Person:
    """人物姿态数据"""
    x_center: float  # 用于排序
    left_wrist: Tuple[float, float]
    right_wrist: Tuple[float, float]
    left_shoulder: Tuple[float, float]
    right_shoulder: Tuple[float, float]
    person_id: str  # 'A', 'B', 'C'

    def is_hands_raised(self) -> bool:
        """判断是否举手（两只手都高于肩膀）"""
        left_raised = self.left_wrist[1] < self.left_shoulder[1]
        right_raised = self.right_wrist[1] < self.right_shoulder[1]
        return left_raised and right_raised


class CircuitDetector:
    """电路闭合检测器（带防抖）"""

    def __init__(self, distance_threshold: float = 0.15, debounce_frames: int = 10):
        self.distance_threshold = distance_threshold
        self.debounce_frames = debounce_frames
        self.closed_count = 0
        self.open_count = 0
        self.current_state = False

    def calculate_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """计算两点欧式距离"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def check_circuit(self, person_a: Person, person_b: Person, person_c: Person) -> Tuple[bool, float, List[float]]:
        """
        检查电路是否闭合
        返回: (是否闭合, 平均距离, [d1, d2, d3])
        """
        # 检查所有人是否举手
        if not (person_a.is_hands_raised() and person_b.is_hands_raised() and person_c.is_hands_raised()):
            return False, 1.0, [1.0, 1.0, 1.0]

        # 计算三对手腕距离
        # A 右手 ↔ B 左手
        d1 = self.calculate_distance(person_a.right_wrist, person_b.left_wrist)
        # B 右手 ↔ C 左手
        d2 = self.calculate_distance(person_b.right_wrist, person_c.left_wrist)
        # C 右手 ↔ A 左手
        d3 = self.calculate_distance(person_c.right_wrist, person_a.left_wrist)

        distances = [d1, d2, d3]
        d_avg = sum(distances) / 3

        # 判断是否所有距离都小于阈值
        all_close = all(d < self.distance_threshold for d in distances)

        return all_close, d_avg, distances

    def update(self, person_a: Person, person_b: Person, person_c: Person) -> Tuple[bool, float, List[float]]:
        """
        更新状态（带防抖）
        返回: (当前稳定状态, 平均距离, [d1, d2, d3])
        """
        instant_closed, d_avg, distances = self.check_circuit(person_a, person_b, person_c)

        if instant_closed:
            self.closed_count += 1
            self.open_count = 0
            if self.closed_count >= self.debounce_frames and not self.current_state:
                self.current_state = True
        else:
            self.open_count += 1
            self.closed_count = 0
            if self.open_count >= self.debounce_frames and self.current_state:
                self.current_state = False

        return self.current_state, d_avg, distances


class AudioController:
    """音频控制器"""

    def __init__(self, drum_path: str, bass_path: str, harmony_path: str):
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

        # 加载音轨
        self.drum = pygame.mixer.Sound(drum_path)
        self.bass = pygame.mixer.Sound(bass_path)
        self.harmony = pygame.mixer.Sound(harmony_path)

        self.tracks = [self.drum, self.bass, self.harmony]
        self.is_playing = False
        self.channels = []

    def start_playback(self):
        """开始播放所有音轨"""
        if not self.is_playing:
            self.channels = []
            for track in self.tracks:
                channel = track.play(loops=-1)  # 循环播放
                self.channels.append(channel)
            self.is_playing = True
            print("🎵 音乐开始播放")

    def stop_playback(self):
        """停止播放所有音轨"""
        if self.is_playing:
            for track in self.tracks:
                track.stop()
            self.channels = []
            self.is_playing = False
            print("🔇 音乐停止")

    def set_volume(self, distance_avg: float, max_distance: float = 0.15):
        """根据平均距离控制音量（距离越近音量越大）"""
        if self.is_playing and self.channels:
            # 距离越小，音量越大
            volume = max(0.0, min(1.0, 1.0 - (distance_avg / max_distance)))
            for channel in self.channels:
                if channel:
                    channel.set_volume(volume)

    def cleanup(self):
        """清理资源"""
        self.stop_playback()
        pygame.mixer.quit()


class VisualFeedback:
    """视觉反馈绘制"""

    @staticmethod
    def draw_person_landmarks(frame, person: Person, color: Tuple[int, int, int], frame_width: int, frame_height: int):
        """绘制人物关键点"""
        # 转换归一化坐标到像素坐标
        points = {
            'left_wrist': person.left_wrist,
            'right_wrist': person.right_wrist,
            'left_shoulder': person.left_shoulder,
            'right_shoulder': person.right_shoulder
        }

        for name, (x, y) in points.items():
            px, py = int(x * frame_width), int(y * frame_height)
            cv2.circle(frame, (px, py), 8, color, -1)

        # 绘制人物标识
        text_x = int(person.x_center * frame_width)
        cv2.putText(frame, person.person_id, (text_x, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    @staticmethod
    def draw_connections(frame, person_a: Person, person_b: Person, person_c: Person,
                        distances: List[float], circuit_closed: bool,
                        frame_width: int, frame_height: int, threshold: float):
        """绘制连接线"""
        connections = [
            (person_a.right_wrist, person_b.left_wrist, distances[0]),
            (person_b.right_wrist, person_c.left_wrist, distances[1]),
            (person_c.right_wrist, person_a.left_wrist, distances[2])
        ]

        for (p1, p2, dist) in connections:
            x1, y1 = int(p1[0] * frame_width), int(p1[1] * frame_height)
            x2, y2 = int(p2[0] * frame_width), int(p2[1] * frame_height)

            # 根据距离和电路状态选择颜色和粗细
            if circuit_closed:
                color = (0, 255, 0)  # 绿色
                thickness = 6
            elif dist < threshold:
                color = (0, 255, 255)  # 黄色
                thickness = 4
            else:
                color = (0, 0, 255)  # 红色
                thickness = 2

            cv2.line(frame, (x1, y1), (x2, y2), color, thickness)

            # 显示距离
            mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.putText(frame, f"{dist:.3f}", (mid_x, mid_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


class BodyCircuitBand:
    """身体回路乐队主类"""

    def __init__(self, drum_path: str, bass_path: str, harmony_path: str):
        # 初始化 MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 初始化组件
        self.circuit_detector = CircuitDetector(distance_threshold=0.15, debounce_frames=10)
        self.audio_controller = AudioController(drum_path, bass_path, harmony_path)
        self.visual = VisualFeedback()

        # 状态
        self.previous_circuit_state = False

    def extract_person_data(self, results, frame_width: int, frame_height: int) -> Optional[Person]:
        """从 MediaPipe 结果提取人物数据"""
        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark

        # 获取关键点
        left_wrist = (landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].x,
                     landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].y)
        right_wrist = (landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].x,
                      landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].y)
        left_shoulder = (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y)
        right_shoulder = (landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y)

        # 计算中心 x 坐标用于排序
        x_center = (left_shoulder[0] + right_shoulder[0]) / 2

        return Person(
            x_center=x_center,
            left_wrist=left_wrist,
            right_wrist=right_wrist,
            left_shoulder=left_shoulder,
            right_shoulder=right_shoulder,
            person_id=""
        )

    def process_frame(self, frame):
        """处理单帧"""
        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 检测姿态（注意：MediaPipe Pose 一次只能检测一个人）
        # 为了简化 demo，我们假设使用多次检测或使用其他方法
        # 这里提供一个简化版本，实际应用中可能需要使用 BlazePose 或其他多人检测方案

        results = self.pose.process(rgb_frame)
        person = self.extract_person_data(results, frame_width, frame_height)

        return person

    def run(self, camera_index: int = 0):
        """运行主循环"""
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print("❌ 无法打开摄像头")
            return

        print("✅ 身体回路乐队已启动")
        print("📝 提示：需要 3 个人同时举手并手拉手形成回路")
        print("按 'q' 退出")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)  # 镜像翻转
                frame_height, frame_width = frame.shape[:2]

                # TODO: 实际应用中需要检测多个人
                # 这里是简化版本的演示框架
                person = self.process_frame(frame)

                # 显示提示信息
                cv2.putText(frame, "Body Circuit Band - Demo", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, "Note: This demo needs 3 people detection", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow('Body Circuit Band', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.audio_controller.cleanup()
            print("👋 程序已退出")


def main():
    # 音频文件路径（需要准备三个音频文件）
    drum_path = "audio_samples/drum.wav"
    bass_path = "audio_samples/bass.wav"
    harmony_path = "audio_samples/harmony.wav"

    band = BodyCircuitBand(drum_path, bass_path, harmony_path)
    band.run(camera_index=0)


if __name__ == "__main__":
    main()
