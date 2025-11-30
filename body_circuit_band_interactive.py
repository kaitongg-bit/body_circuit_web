"""
Body Circuit Band - Interactive Version
Supports any number of people (2, 3, 4, 5, ...).
All people form a circle by holding hands.
"""

import cv2
import numpy as np
import pygame
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass
import colorsys

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed")


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


class CircuitDetectorDynamic:
    """Circuit closure detector for any number of people"""

    def __init__(self, num_people: int, distance_threshold: float = 0.3,
                 debounce_frames_close: int = 8, debounce_frames_open: int = 2):
        self.num_people = num_people
        self.distance_threshold = distance_threshold
        self.debounce_frames_close = debounce_frames_close
        self.debounce_frames_open = debounce_frames_open
        self.closed_count = 0
        self.open_count = 0
        self.current_state = False

    def calculate_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def check_circuit(self, persons: List[Person]) -> Tuple[bool, float, List[float]]:
        """
        Check if circuit is closed for N people
        Circuit: Each person's right hand connects to next person's left hand (forming a circle)
        Returns: (is_closed, average_distance, [d1, d2, ..., dN])
        """
        distances = []

        for i in range(self.num_people):
            next_i = (i + 1) % self.num_people
            d = self.calculate_distance(persons[i].right_wrist, persons[next_i].left_wrist)
            distances.append(d)

        d_avg = sum(distances) / len(distances)
        all_close = all(d < self.distance_threshold for d in distances)

        return all_close, d_avg, distances

    def update(self, persons: List[Person]) -> Tuple[bool, float, List[float]]:
        """
        Update state with asymmetric debouncing (slow to close, fast to open)
        Returns: (current_stable_state, average_distance, [d1, d2, ..., dN])
        """
        instant_closed, d_avg, distances = self.check_circuit(persons)

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
                    channel.set_volume(0.5)
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

    def set_volume(self, distance_avg: float, max_distance: float = 0.3):
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
    """Visual feedback rendering with dynamic color generation"""

    def __init__(self, num_people: int):
        self.num_people = num_people
        self.colors = self._generate_colors(num_people)

    def _generate_colors(self, n: int) -> dict:
        """Generate N distinct colors using HSV color space"""
        colors = {}
        labels = [chr(65 + i) for i in range(min(n, 26))]  # A, B, C, ..., Z

        for i, label in enumerate(labels):
            hue = i / n
            rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            colors[label] = bgr

        return colors

    def draw_person_landmarks(self, frame, person: Person, frame_width: int, frame_height: int):
        """Draw person keypoints"""
        color = self.colors.get(person.person_id, (255, 255, 255))

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

    def draw_connections(self, frame, persons: List[Person],
                        distances: List[float], circuit_closed: bool,
                        frame_width: int, frame_height: int, threshold: float):
        """Draw connection lines for all people in a circle"""
        for i in range(len(persons)):
            next_i = (i + 1) % len(persons)
            p1 = persons[i].right_wrist
            p2 = persons[next_i].left_wrist
            dist = distances[i]

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
            label = f"{persons[i].person_id}->{persons[next_i].person_id}"
            cv2.putText(frame, f"{label}: {dist:.3f}", (mid_x - 40, mid_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    @staticmethod
    def draw_status_panel(frame, circuit_closed: bool, d_avg: float, volume: float, num_people: int):
        """Draw status panel"""
        panel_height = 150
        panel = np.zeros((panel_height, frame.shape[1], 3), dtype=np.uint8)

        if circuit_closed:
            panel[:, :] = (0, 50, 0)
        else:
            panel[:, :] = (50, 0, 0)

        status_text = f"CIRCUIT CLOSED - MUSIC PLAYING ({num_people} people)" if circuit_closed else f"CIRCUIT OPEN ({num_people} people)"
        cv2.putText(panel, status_text, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

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


class BodyCircuitBandInteractive:
    """Body Circuit Band with customizable number of people"""

    def __init__(self, num_people: int, drum_path: str, bass_path: str, harmony_path: str, use_yolo: bool = True):
        if num_people < 2:
            raise ValueError("Number of people must be at least 2")

        self.num_people = num_people
        self.use_yolo = use_yolo and YOLO_AVAILABLE

        if self.use_yolo:
            print(f"Using YOLOv8-Pose for {num_people}-person detection")
            self.model = YOLO('yolov8n-pose.pt')
        else:
            print("Using simplified detection mode (demo only)")
            self.model = None

        self.circuit_detector = CircuitDetectorDynamic(
            num_people=num_people,
            distance_threshold=0.3,
            debounce_frames_close=8,
            debounce_frames_open=2
        )
        self.audio_controller = AudioController(drum_path, bass_path, harmony_path)
        self.visual = VisualFeedback(num_people)

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

        print(f"\nBody Circuit Band ({self.num_people} People) Started")
        print(f"Note: Need {self.num_people} people to form a circuit")
        print("Circuit formation (forming a circle):")
        for i in range(self.num_people):
            next_i = (i + 1) % self.num_people
            person_label = chr(65 + i)
            next_label = chr(65 + next_i)
            print(f"   - Person {person_label} right hand <-> Person {next_label} left hand")
        print("   - Hands don't need to be raised - just connected!")
        print("Press 'q' to quit\n")

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

                if len(persons) >= self.num_people:
                    # Assign person IDs
                    for i in range(self.num_people):
                        persons[i].person_id = chr(65 + i)  # A, B, C, ...

                    circuit_closed, d_avg, distances = self.circuit_detector.update(
                        persons[:self.num_people]
                    )

                    # Debug output
                    dist_str = ", ".join([f"{chr(65+i)}->{chr(65+(i+1)%self.num_people)}={distances[i]:.3f}"
                                         for i in range(self.num_people)])
                    print(f"Distances: {dist_str}, avg={d_avg:.3f}")
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

                    for person in persons[:self.num_people]:
                        self.visual.draw_person_landmarks(frame, person, frame_width, frame_height)

                    self.visual.draw_connections(
                        frame, persons[:self.num_people],
                        distances, circuit_closed, frame_width, frame_height,
                        self.circuit_detector.distance_threshold
                    )

                    volume = max(0.0, min(1.0, 1.0 - (d_avg / 0.3))) if circuit_closed else 0.0

                    frame = self.visual.draw_status_panel(frame, circuit_closed, d_avg, volume, self.num_people)

                else:
                    cv2.putText(frame, f"Detected: {len(persons)}/{self.num_people} people", (20, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    cv2.putText(frame, f"Need {self.num_people} people to start!", (20, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

                cv2.imshow(f'Body Circuit Band - {self.num_people} People', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.audio_controller.cleanup()
            print("Program exited")


def main():
    """Main entry point with interactive input"""
    print("=" * 60)
    print("Body Circuit Band - Interactive Version")
    print("=" * 60)
    print("\nThis program supports any number of people forming a circuit!")
    print("All people stand in a row and will be arranged in a circle.")
    print("Each person's right hand connects to the next person's left hand.\n")

    while True:
        try:
            num_people = int(input("How many people will participate? (2 or more): "))
            if num_people >= 2:
                break
            else:
                print("Please enter a number >= 2")
        except ValueError:
            print("Please enter a valid number")

    print(f"\nGreat! Starting {num_people}-person mode...")
    print("Loading audio files and initializing camera...\n")

    drum_path = "audio_samples_v2/drum.wav"
    bass_path = "audio_samples_v2/bass.wav"
    harmony_path = "audio_samples_v2/harmony.wav"

    band = BodyCircuitBandInteractive(num_people, drum_path, bass_path, harmony_path, use_yolo=True)
    band.run(camera_index=0)


if __name__ == "__main__":
    main()
