"""
Body Circuit Band - Full Version
Interactive music installation using YOLOv8 Pose for multi-person pose recognition.
"""

import cv2
import numpy as np
import pygame
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed, will use simplified mode")


@dataclass
class Person:
    """Person pose data"""
    x_center: float
    left_wrist: Tuple[float, float]
    right_wrist: Tuple[float, float]
    left_shoulder: Tuple[float, float]
    right_shoulder: Tuple[float, float]
    person_id: str

    def is_hands_raised(self) -> bool:
        """Check if both hands are raised above shoulders"""
        left_raised = self.left_wrist[1] < self.left_shoulder[1]
        right_raised = self.right_wrist[1] < self.right_shoulder[1]
        return left_raised and right_raised


class CircuitDetector:
    """Circuit closure detector with asymmetric debouncing"""

    def __init__(self, distance_threshold: float = 0.15, debounce_frames_close: int = 10, debounce_frames_open: int = 3):
        self.distance_threshold = distance_threshold
        self.debounce_frames_close = debounce_frames_close
        self.debounce_frames_open = debounce_frames_open
        self.closed_count = 0
        self.open_count = 0
        self.current_state = False

    def calculate_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def check_circuit(self, person_a: Person, person_b: Person, person_c: Person) -> Tuple[bool, float, List[float]]:
        """
        Check if circuit is closed
        Returns: (is_closed, average_distance, [d1, d2, d3])
        """
        # Removed hand-raising requirement - just check if hands are connected
        d1 = self.calculate_distance(person_a.right_wrist, person_b.left_wrist)
        d2 = self.calculate_distance(person_b.right_wrist, person_c.left_wrist)
        d3 = self.calculate_distance(person_c.right_wrist, person_a.left_wrist)

        distances = [d1, d2, d3]
        d_avg = sum(distances) / 3

        all_close = all(d < self.distance_threshold for d in distances)

        return all_close, d_avg, distances

    def update(self, person_a: Person, person_b: Person, person_c: Person) -> Tuple[bool, float, List[float]]:
        """
        Update state with asymmetric debouncing (slow to close, fast to open)
        Returns: (current_stable_state, average_distance, [d1, d2, d3])
        """
        instant_closed, d_avg, distances = self.check_circuit(person_a, person_b, person_c)

        if instant_closed:
            self.closed_count += 1
            self.open_count = 0
            if self.closed_count >= self.debounce_frames_close and not self.current_state:
                self.current_state = True
                print(f"Circuit closed! (consecutive frames: {self.closed_count})")
        else:
            self.open_count += 1
            self.closed_count = 0
            if self.open_count >= self.debounce_frames_open and self.current_state:
                self.current_state = False
                print(f"Circuit opened (consecutive frames: {self.open_count})")

        return self.current_state, d_avg, distances


class AudioController:
    """Audio controller"""

    def __init__(self, drum_path: str, bass_path: str, harmony_path: str):
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

        try:
            self.drum = pygame.mixer.Sound(drum_path)
            self.bass = pygame.mixer.Sound(bass_path)
            self.harmony = pygame.mixer.Sound(harmony_path)
            self.tracks = [self.drum, self.bass, self.harmony]
            self.audio_available = True
        except Exception as e:
            print(f"Audio file loading failed: {e}")
            self.audio_available = False
            self.tracks = []

        self.is_playing = False
        self.channels = []

    def start_playback(self):
        """Start playing all tracks"""
        if not self.audio_available:
            return

        if not self.is_playing:
            self.channels = []
            for track in self.tracks:
                channel = track.play(loops=-1)
                if channel:
                    channel.set_volume(0.5)  # Set initial volume to 50%
                self.channels.append(channel)
            self.is_playing = True
            print("Music started playing")

    def stop_playback(self):
        """Stop playing all tracks"""
        if not self.audio_available:
            return

        if self.is_playing:
            for track in self.tracks:
                track.stop()
            self.channels = []
            self.is_playing = False
            print("Music stopped")

    def set_volume(self, distance_avg: float, max_distance: float = 0.15):
        """Control volume based on average distance (closer = louder)"""
        if not self.audio_available:
            return

        if self.is_playing and self.channels:
            volume = max(0.0, min(1.0, 1.0 - (distance_avg / max_distance)))
            for channel in self.channels:
                if channel:
                    channel.set_volume(volume)

    def cleanup(self):
        """Clean up resources"""
        self.stop_playback()
        if self.audio_available:
            pygame.mixer.quit()


class VisualFeedback:
    """Visual feedback rendering"""

    COLORS = {
        'A': (255, 0, 0),
        'B': (0, 255, 0),
        'C': (0, 0, 255)
    }

    @staticmethod
    def draw_person_landmarks(frame, person: Person, frame_width: int, frame_height: int):
        """Draw person keypoints"""
        color = VisualFeedback.COLORS.get(person.person_id, (255, 255, 255))

        points = {
            'left_wrist': person.left_wrist,
            'right_wrist': person.right_wrist,
            'left_shoulder': person.left_shoulder,
            'right_shoulder': person.right_shoulder
        }

        for name, (x, y) in points.items():
            px, py = int(x * frame_width), int(y * frame_height)
            cv2.circle(frame, (px, py), 10, color, -1)
            cv2.circle(frame, (px, py), 12, (255, 255, 255), 2)

        ls_px, ls_py = int(person.left_shoulder[0] * frame_width), int(person.left_shoulder[1] * frame_height)
        rs_px, rs_py = int(person.right_shoulder[0] * frame_width), int(person.right_shoulder[1] * frame_height)
        cv2.line(frame, (ls_px, ls_py), (rs_px, rs_py), color, 3)

        lw_px, lw_py = int(person.left_wrist[0] * frame_width), int(person.left_wrist[1] * frame_height)
        rw_px, rw_py = int(person.right_wrist[0] * frame_width), int(person.right_wrist[1] * frame_height)
        cv2.line(frame, (ls_px, ls_py), (lw_px, lw_py), color, 3)
        cv2.line(frame, (rs_px, rs_py), (rw_px, rw_py), color, 3)

        text_x = int(person.x_center * frame_width)
        text_y = ls_py - 30
        cv2.putText(frame, f"Person {person.person_id}", (text_x - 50, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        status = "Hands Up" if person.is_hands_raised() else "Hands Down"
        status_color = (0, 255, 0) if person.is_hands_raised() else (0, 0, 255)
        cv2.putText(frame, status, (text_x - 50, text_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    @staticmethod
    def draw_connections(frame, person_a: Person, person_b: Person, person_c: Person,
                        distances: List[float], circuit_closed: bool,
                        frame_width: int, frame_height: int, threshold: float):
        """Draw connection lines"""
        connections = [
            (person_a.right_wrist, person_b.left_wrist, distances[0], "A->B"),
            (person_b.right_wrist, person_c.left_wrist, distances[1], "B->C"),
            (person_c.right_wrist, person_a.left_wrist, distances[2], "C->A")
        ]

        for (p1, p2, dist, label) in connections:
            x1, y1 = int(p1[0] * frame_width), int(p1[1] * frame_height)
            x2, y2 = int(p2[0] * frame_width), int(p2[1] * frame_height)

            if circuit_closed:
                color = (0, 255, 0)
                thickness = 8
            elif dist < threshold:
                color = (0, 255, 255)
                thickness = 5
            else:
                color = (0, 0, 255)
                thickness = 3

            cv2.line(frame, (x1, y1), (x2, y2), color, thickness)

            mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.putText(frame, f"{label}: {dist:.3f}", (mid_x - 40, mid_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    @staticmethod
    def draw_status_panel(frame, circuit_closed: bool, d_avg: float, volume: float):
        """Draw status panel"""
        panel_height = 150
        panel = np.zeros((panel_height, frame.shape[1], 3), dtype=np.uint8)

        if circuit_closed:
            panel[:, :] = (0, 50, 0)
        else:
            panel[:, :] = (50, 0, 0)

        status_text = "CIRCUIT CLOSED - MUSIC PLAYING" if circuit_closed else "CIRCUIT OPEN"
        cv2.putText(panel, status_text, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        cv2.putText(panel, f"Avg Distance: {d_avg:.4f}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if circuit_closed:
            cv2.putText(panel, f"Volume: {volume:.0%}", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            bar_x, bar_y = 300, 100
            bar_width, bar_height = 400, 30
            cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                         (255, 255, 255), 2)
            fill_width = int(bar_width * volume)
            cv2.rectangle(panel, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height),
                         (0, 255, 0), -1)

        return np.vstack([panel, frame])


class BodyCircuitBand:
    """Body Circuit Band main class"""

    def __init__(self, drum_path: str, bass_path: str, harmony_path: str, use_yolo: bool = True):
        self.use_yolo = use_yolo and YOLO_AVAILABLE
        if self.use_yolo:
            print("Using YOLOv8-Pose for multi-person detection")
            self.model = YOLO('yolov8n-pose.pt')
        else:
            print("Using simplified detection mode (demo only)")
            self.model = None

        self.circuit_detector = CircuitDetector(distance_threshold=0.3, debounce_frames_close=8, debounce_frames_open=2)
        self.audio_controller = AudioController(drum_path, bass_path, harmony_path)
        self.visual = VisualFeedback()

        self.previous_circuit_state = False

    def extract_persons_yolo(self, results, frame_width: int, frame_height: int) -> List[Person]:
        """Extract multiple persons from YOLO results"""
        persons = []

        if len(results) == 0 or results[0].keypoints is None:
            return persons

        keypoints_data = results[0].keypoints.data

        for person_kps in keypoints_data:
            left_shoulder = (person_kps[5][0].item() / frame_width, person_kps[5][1].item() / frame_height)
            right_shoulder = (person_kps[6][0].item() / frame_width, person_kps[6][1].item() / frame_height)
            left_wrist = (person_kps[9][0].item() / frame_width, person_kps[9][1].item() / frame_height)
            right_wrist = (person_kps[10][0].item() / frame_width, person_kps[10][1].item() / frame_height)

            if (person_kps[5][2] < 0.5 or person_kps[6][2] < 0.5 or
                person_kps[9][2] < 0.5 or person_kps[10][2] < 0.5):
                continue

            x_center = (left_shoulder[0] + right_shoulder[0]) / 2

            person = Person(
                x_center=x_center,
                left_wrist=left_wrist,
                right_wrist=right_wrist,
                left_shoulder=left_shoulder,
                right_shoulder=right_shoulder,
                person_id=""
            )
            persons.append(person)

        return persons

    def run(self, camera_index: int = 0):
        """Run main loop"""
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print("Unable to open camera")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("Body Circuit Band Started")
        print("Note: Need 3 people to form a circuit by connecting hands")
        print("   - A right hand <-> B left hand")
        print("   - B right hand <-> C left hand")
        print("   - C right hand <-> A left hand")
        print("   - Hands don't need to be raised - just connected!")
        print("Press 'q' to quit")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                frame_height, frame_width = frame.shape[:2]

                if self.use_yolo:
                    results = self.model(frame, verbose=False)
                    persons = self.extract_persons_yolo(results, frame_width, frame_height)
                else:
                    persons = []

                persons.sort(key=lambda p: p.x_center)

                if len(persons) >= 3:
                    persons[0].person_id = 'A'
                    persons[1].person_id = 'B'
                    persons[2].person_id = 'C'

                    circuit_closed, d_avg, distances = self.circuit_detector.update(
                        persons[0], persons[1], persons[2]
                    )

                    # Debug output
                    hands_status = [p.is_hands_raised() for p in persons[:3]]
                    print(f"Hands raised: A={hands_status[0]}, B={hands_status[1]}, C={hands_status[2]}")
                    print(f"Distances: A->B={distances[0]:.3f}, B->C={distances[1]:.3f}, C->A={distances[2]:.3f}, avg={d_avg:.3f}")
                    print(f"Circuit closed: {circuit_closed}, Threshold: {self.circuit_detector.distance_threshold}")

                    if circuit_closed and not self.previous_circuit_state:
                        print(">>> STARTING MUSIC <<<")
                        self.audio_controller.start_playback()
                    elif not circuit_closed and self.previous_circuit_state:
                        print(">>> STOPPING MUSIC <<<")
                        self.audio_controller.stop_playback()

                    if circuit_closed:
                        self.audio_controller.set_volume(d_avg, max_distance=0.3)

                    self.previous_circuit_state = circuit_closed

                    for person in persons[:3]:
                        self.visual.draw_person_landmarks(frame, person, frame_width, frame_height)

                    self.visual.draw_connections(
                        frame, persons[0], persons[1], persons[2],
                        distances, circuit_closed, frame_width, frame_height,
                        self.circuit_detector.distance_threshold
                    )

                    volume = max(0.0, min(1.0, 1.0 - (d_avg / 0.3))) if circuit_closed else 0.0

                    frame = self.visual.draw_status_panel(frame, circuit_closed, d_avg, volume)

                else:
                    cv2.putText(frame, f"Detected: {len(persons)}/3 people", (20, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    cv2.putText(frame, "Need 3 people to start!", (20, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

                cv2.imshow('Body Circuit Band', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.audio_controller.cleanup()
            print("Program exited")


def main():
    """Main entry point"""
    drum_path = "audio_samples_v2/drum.wav"
    bass_path = "audio_samples_v2/bass.wav"
    harmony_path = "audio_samples_v2/harmony.wav"

    band = BodyCircuitBand(drum_path, bass_path, harmony_path, use_yolo=True)
    band.run(camera_index=0)


if __name__ == "__main__":
    main()
