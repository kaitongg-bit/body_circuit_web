"""
身体回路乐队 (Body Circuit Band) - 简化版
使用 MediaPipe Pose 的简化演示版本
注意：MediaPipe Pose 单次只能检测一个人，这个版本仅用于演示核心逻辑
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
    x_center: float
    left_wrist: Tuple[float, float]
    right_wrist: Tuple[float, float]
    left_shoulder: Tuple[float, float]
    right_shoulder: Tuple[float, float]
    person_id: str

    def is_hands_raised(self) -> bool:
        """判断是否举手"""
        left_raised = self.left_wrist[1] < self.left_shoulder[1]
        right_raised = self.right_wrist[1] < self.right_shoulder[1]
        return left_raised and right_raised


class DemoMode:
    """演示模式：模拟 3 个人的互动"""

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 模拟三个人的位置偏移
        self.offsets = [
            (-0.25, 0),  # Person A (左侧)
            (0, 0),      # Person B (中间)
            (0.25, 0)    # Person C (右侧)
        ]

    def create_demo_persons(self, base_person: Person) -> List[Person]:
        """基于检测到的一个人创建三个模拟人物"""
        persons = []

        for i, (offset_x, offset_y) in enumerate(self.offsets):
            person = Person(
                x_center=base_person.x_center + offset_x,
                left_wrist=(base_person.left_wrist[0] + offset_x,
                           base_person.left_wrist[1] + offset_y),
                right_wrist=(base_person.right_wrist[0] + offset_x,
                            base_person.right_wrist[1] + offset_y),
                left_shoulder=(base_person.left_shoulder[0] + offset_x,
                              base_person.left_shoulder[1] + offset_y),
                right_shoulder=(base_person.right_shoulder[0] + offset_x,
                               base_person.right_shoulder[1] + offset_y),
                person_id=chr(65 + i)  # 'A', 'B', 'C'
            )
            persons.append(person)

        return persons

    def draw_person(self, frame, person: Person, color, frame_width, frame_height):
        """绘制人物"""
        # 绘制关键点
        points = [
            person.left_wrist, person.right_wrist,
            person.left_shoulder, person.right_shoulder
        ]

        for x, y in points:
            px, py = int(x * frame_width), int(y * frame_height)
            if 0 <= px < frame_width and 0 <= py < frame_height:
                cv2.circle(frame, (px, py), 8, color, -1)

        # 绘制骨架
        ls = (int(person.left_shoulder[0] * frame_width),
              int(person.left_shoulder[1] * frame_height))
        rs = (int(person.right_shoulder[0] * frame_width),
              int(person.right_shoulder[1] * frame_height))
        lw = (int(person.left_wrist[0] * frame_width),
              int(person.left_wrist[1] * frame_height))
        rw = (int(person.right_wrist[0] * frame_width),
              int(person.right_wrist[1] * frame_height))

        if all(0 <= p[0] < frame_width and 0 <= p[1] < frame_height
               for p in [ls, rs, lw, rw]):
            cv2.line(frame, ls, rs, color, 3)
            cv2.line(frame, ls, lw, color, 3)
            cv2.line(frame, rs, rw, color, 3)

        # 标签
        label_x = int(person.x_center * frame_width)
        label_y = int(person.left_shoulder[1] * frame_height) - 30
        if 0 <= label_x < frame_width and 0 <= label_y < frame_height:
            cv2.putText(frame, f"Person {person.person_id}", (label_x - 50, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def draw_connections(self, frame, persons, distances, circuit_closed, frame_width, frame_height):
        """绘制连接线"""
        connections = [
            (persons[0].right_wrist, persons[1].left_wrist, distances[0]),
            (persons[1].right_wrist, persons[2].left_wrist, distances[1]),
            (persons[2].right_wrist, persons[0].left_wrist, distances[2])
        ]

        for p1, p2, dist in connections:
            x1, y1 = int(p1[0] * frame_width), int(p1[1] * frame_height)
            x2, y2 = int(p2[0] * frame_width), int(p2[1] * frame_height)

            if circuit_closed:
                color = (0, 255, 0)
                thickness = 6
            elif dist < 0.15:
                color = (0, 255, 255)
                thickness = 4
            else:
                color = (0, 0, 255)
                thickness = 2

            if (0 <= x1 < frame_width and 0 <= y1 < frame_height and
                0 <= x2 < frame_width and 0 <= y2 < frame_height):
                cv2.line(frame, (x1, y1), (x2, y2), color, thickness)


class SimpleCircuitBand:
    """简化版身体回路乐队"""

    def __init__(self):
        self.demo = DemoMode()
        self.circuit_closed = False
        self.closed_count = 0
        self.open_count = 0

        # 初始化音频
        try:
            pygame.mixer.init()
            self.drum = pygame.mixer.Sound("audio_samples/drum.wav")
            self.bass = pygame.mixer.Sound("audio_samples/bass.wav")
            self.harmony = pygame.mixer.Sound("audio_samples/harmony.wav")
            self.audio_available = True
        except:
            print("⚠️  音频文件未找到，音频功能将被禁用")
            self.audio_available = False

        self.is_playing = False

    def calculate_distance(self, p1, p2):
        """计算距离"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def check_circuit(self, persons):
        """检查电路"""
        if len(persons) < 3:
            return False, 1.0, [1.0, 1.0, 1.0]

        # 检查举手
        if not all(p.is_hands_raised() for p in persons[:3]):
            return False, 1.0, [1.0, 1.0, 1.0]

        # 计算距离
        d1 = self.calculate_distance(persons[0].right_wrist, persons[1].left_wrist)
        d2 = self.calculate_distance(persons[1].right_wrist, persons[2].left_wrist)
        d3 = self.calculate_distance(persons[2].right_wrist, persons[0].left_wrist)

        distances = [d1, d2, d3]
        d_avg = sum(distances) / 3

        all_close = all(d < 0.15 for d in distances)

        return all_close, d_avg, distances

    def update_circuit(self, persons):
        """更新电路状态（带防抖）"""
        instant_closed, d_avg, distances = self.check_circuit(persons)

        if instant_closed:
            self.closed_count += 1
            self.open_count = 0
            if self.closed_count >= 10 and not self.circuit_closed:
                self.circuit_closed = True
                if self.audio_available:
                    self.drum.play(loops=-1)
                    self.bass.play(loops=-1)
                    self.harmony.play(loops=-1)
                    self.is_playing = True
                print("✅ 电路闭合！音乐播放")
        else:
            self.open_count += 1
            self.closed_count = 0
            if self.open_count >= 10 and self.circuit_closed:
                self.circuit_closed = False
                if self.audio_available:
                    self.drum.stop()
                    self.bass.stop()
                    self.harmony.stop()
                    self.is_playing = False
                print("❌ 电路断开！音乐停止")

        return d_avg, distances

    def run(self):
        """运行主循环"""
        cap = cv2.VideoCapture(0)

        print("✅ 简化版身体回路乐队已启动")
        print("📝 提示：举起双手，程序将模拟 3 个人的互动")
        print("   实际检测到的是你的动作，画面会显示 3 个模拟人物")
        print("按 'q' 退出")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                frame_height, frame_width = frame.shape[:2]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 检测姿态
                results = self.demo.pose.process(rgb_frame)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    # 提取关键点
                    base_person = Person(
                        x_center=(landmarks[self.demo.mp_pose.PoseLandmark.LEFT_SHOULDER].x +
                                 landmarks[self.demo.mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2,
                        left_wrist=(landmarks[self.demo.mp_pose.PoseLandmark.LEFT_WRIST].x,
                                   landmarks[self.demo.mp_pose.PoseLandmark.LEFT_WRIST].y),
                        right_wrist=(landmarks[self.demo.mp_pose.PoseLandmark.RIGHT_WRIST].x,
                                    landmarks[self.demo.mp_pose.PoseLandmark.RIGHT_WRIST].y),
                        left_shoulder=(landmarks[self.demo.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                      landmarks[self.demo.mp_pose.PoseLandmark.LEFT_SHOULDER].y),
                        right_shoulder=(landmarks[self.demo.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                                       landmarks[self.demo.mp_pose.PoseLandmark.RIGHT_SHOULDER].y),
                        person_id='B'
                    )

                    # 创建模拟的三个人
                    persons = self.demo.create_demo_persons(base_person)

                    # 更新电路状态
                    d_avg, distances = self.update_circuit(persons)

                    # 绘制
                    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
                    for person, color in zip(persons, colors):
                        self.demo.draw_person(frame, person, color, frame_width, frame_height)

                    self.demo.draw_connections(frame, persons, distances, self.circuit_closed,
                                              frame_width, frame_height)

                    # 状态显示
                    status = "🎵 CIRCUIT CLOSED" if self.circuit_closed else "⭕ CIRCUIT OPEN"
                    color = (0, 255, 0) if self.circuit_closed else (0, 0, 255)
                    cv2.putText(frame, status, (20, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                    cv2.putText(frame, f"Avg Distance: {d_avg:.3f}", (20, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                else:
                    cv2.putText(frame, "No person detected", (20, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

                cv2.putText(frame, "DEMO MODE - Simulating 3 people", (20, frame_height - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow('Body Circuit Band - Simple Demo', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.audio_available:
                pygame.mixer.quit()
            print("👋 程序已退出")


if __name__ == "__main__":
    band = SimpleCircuitBand()
    band.run()
