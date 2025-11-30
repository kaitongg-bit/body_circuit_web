"""
Body Circuit Band - Solo Test Mode
Uses YOLOv8-Pose to detect 1 person and replicates to 3 positions to simulate full interaction.
Core logic is identical to the full version to ensure test validity.
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


class CircuitDetector:
    """Circuit closure detector with debouncing"""

    def __init__(self, distance_threshold: float = 0.15, debounce_frames: int = 10):
        self.distance_threshold = distance_threshold
        self.debounce_frames = debounce_frames
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
        if not (person_a.is_hands_raised() and person_b.is_hands_raised() and person_c.is_hands_raised()):
            return False, 1.0, [1.0, 1.0, 1.0]

        d1 = self.calculate_distance(person_a.right_wrist, person_b.left_wrist)
        d2 = self.calculate_distance(person_b.right_wrist, person_c.left_wrist)
        d3 = self.calculate_distance(person_c.right_wrist, person_a.left_wrist)

        distances = [d1, d2, d3]
        d_avg = sum(distances) / 3

        all_close = all(d < self.distance_threshold for d in distances)

        return all_close, d_avg, distances

    def update(self, person_a: Person, person_b: Person, person_c: Person) -> Tuple[bool, float, List[float]]:
        """
        Update state with debouncing
        Returns: (current_stable_state, average_distance, [d1, d2, d3])
        """
        instant_closed, d_avg, distances = self.check_circuit(person_a, person_b, person_c)

        if instant_closed:
            self.closed_count += 1
            self.open_count = 0
            if self.closed_count >= self.debounce_frames and not self.current_state:
                self.current_state = True
                print(f"Circuit closed! (consecutive frames: {self.closed_count})")
        else:
            self.open_count += 1
            self.closed_count = 0
            if self.open_count >= self.debounce_frames and self.current_state:
                self.current_state = False
                print(f"Circuit opened (consecutive frames: {self.open_count})")

        return self.current_state, d_avg, distances


class AudioController:
    """Audio controller with layered playback"""

    def __init__(self, drum_path: str, bass_path: str, harmony_path: str):
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

        try:
            self.drum = pygame.mixer.Sound(drum_path)
            self.bass = pygame.mixer.Sound(bass_path)
            self.harmony = pygame.mixer.Sound(harmony_path)
            self.tracks = [self.drum, self.bass, self.harmony]
            self.track_names = ['Drums', 'Bass', 'Harmony']
            self.audio_available = True
        except Exception as e:
            print(f"Audio file loading failed: {e}")
            self.audio_available = False
            self.tracks = []
            self.track_names = []

        self.is_playing = False
        self.channels = []
        self.current_layer = 0

    def start_playback(self):
        """Start playing all tracks with initial volume at 0"""
        if not self.audio_available:
            return

        if not self.is_playing:
            self.channels = []
            for track in self.tracks:
                channel = track.play(loops=-1)
                if channel:
                    channel.set_volume(0.0)
                self.channels.append(channel)
            self.is_playing = True
            self.current_layer = 0
            print("Music system started (layered mode)")

    def stop_playback(self):
        """Stop playing all tracks"""
        if not self.audio_available:
            return

        if self.is_playing:
            for track in self.tracks:
                track.stop()
            self.channels = []
            self.is_playing = False
            self.current_layer = 0
            print("Music stopped")

    def set_volume_layered(self, distance_avg: float, max_distance: float = 0.15):
        """Layered volume control: closer distance activates more tracks with higher volume"""
        if not self.audio_available or not self.is_playing:
            return

        if not self.channels:
            return

        base_volume = max(0.0, min(1.0, 1.0 - (distance_avg / max_distance)))

        new_layer = 0
        if distance_avg < max_distance * 0.40:
            new_layer = 3
        elif distance_avg < max_distance * 0.75:
            new_layer = 2
        else:
            new_layer = 1

        if new_layer != self.current_layer:
            active_tracks = ', '.join(self.track_names[:new_layer])
            print(f"Music intensity changed: {self.current_layer} layers -> {new_layer} layers ({active_tracks})")
            self.current_layer = new_layer

        for i, channel in enumerate(self.channels):
            if channel:
                if i < new_layer:
                    channel.set_volume(base_volume)
                else:
                    channel.set_volume(0.0)

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
    def draw_status_panel(frame, circuit_closed: bool, d_avg: float, volume: float, is_solo_mode: bool = True):
        """Draw status panel"""
        panel_height = 180
        panel = np.zeros((panel_height, frame.shape[1], 3), dtype=np.uint8)

        if circuit_closed:
            panel[:, :] = (0, 50, 0)
        else:
            panel[:, :] = (50, 0, 0)

        if is_solo_mode:
            cv2.putText(panel, "[SOLO TEST MODE]", (20, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        status_text = "CIRCUIT CLOSED - MUSIC PLAYING" if circuit_closed else "CIRCUIT OPEN"
        cv2.putText(panel, status_text, (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        cv2.putText(panel, f"Avg Distance: {d_avg:.4f}", (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if circuit_closed:
            cv2.putText(panel, f"Volume: {volume:.0%}", (20, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            bar_x, bar_y = 300, 120
            bar_width, bar_height = 400, 30
            cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                         (255, 255, 255), 2)
            fill_width = int(bar_width * volume)
            cv2.rectangle(panel, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height),
                         (0, 255, 0), -1)

        return np.vstack([panel, frame])


class SoloTestMode:
    """Solo test mode: replicates one person's pose to 3 positions with dynamic distance based on stability"""

    def __init__(self):
        self.base_offset = 0.12
        self.wrist_history = []
        self.history_size = 10

    def calculate_stability(self, person: Person) -> float:
        """
        Calculate pose stability based on wrist position changes
        Returns: 0.0 (very unstable) to 1.0 (completely still)
        """
        current_pos = (
            (person.left_wrist[0] + person.right_wrist[0]) / 2,
            (person.left_wrist[1] + person.right_wrist[1]) / 2
        )

        self.wrist_history.append(current_pos)

        if len(self.wrist_history) > self.history_size:
            self.wrist_history.pop(0)

        if len(self.wrist_history) < 3:
            return 0.5

        import numpy as np
        positions = np.array(self.wrist_history)
        std_dev = np.std(positions, axis=0).mean()

        stability = max(0.0, min(1.0, 1.0 - (std_dev / 0.05)))

        return stability

    def create_virtual_persons(self, detected_person: Person) -> tuple:
        """
        Create 3 virtual persons from detected person
        Offset dynamically adjusted based on stability
        Returns: (list of virtual persons, stability score)
        """
        stability = self.calculate_stability(detected_person)

        dynamic_offset = self.base_offset + (0.04 * (1.0 - stability))

        offsets = [
            (-dynamic_offset, 0),
            (0, 0),
            (dynamic_offset, 0)
        ]

        virtual_persons = []

        for i, (offset_x, offset_y) in enumerate(offsets):
            person_id = ['A', 'B', 'C'][i]

            person = Person(
                x_center=detected_person.x_center + offset_x,
                left_wrist=(detected_person.left_wrist[0] + offset_x,
                           detected_person.left_wrist[1] + offset_y),
                right_wrist=(detected_person.right_wrist[0] + offset_x,
                            detected_person.right_wrist[1] + offset_y),
                left_shoulder=(detected_person.left_shoulder[0] + offset_x,
                              detected_person.left_shoulder[1] + offset_y),
                right_shoulder=(detected_person.right_shoulder[0] + offset_x,
                               detected_person.right_shoulder[1] + offset_y),
                person_id=person_id
            )
            virtual_persons.append(person)

        return virtual_persons, stability


class BodyCircuitBandSolo:
    """Body Circuit Band - Solo Test Version"""

    def __init__(self, drum_path: str, bass_path: str, harmony_path: str):
        if not YOLO_AVAILABLE:
            raise RuntimeError("ultralytics is required: pip install ultralytics")

        print("Using YOLOv8-Pose for single person detection")
        self.model = YOLO('yolov8n-pose.pt')

        self.circuit_detector = CircuitDetector(distance_threshold=0.30, debounce_frames=10)
        self.audio_controller = AudioController(drum_path, bass_path, harmony_path)
        self.visual = VisualFeedback()
        self.solo_mode = SoloTestMode()

        self.previous_circuit_state = False

    def extract_person_yolo(self, results, frame_width: int, frame_height: int) -> Optional[Person]:
        """Extract first detected person from YOLO results"""
        if len(results) == 0 or results[0].keypoints is None:
            return None

        keypoints_data = results[0].keypoints.data

        if len(keypoints_data) == 0:
            return None

        person_kps = keypoints_data[0]

        left_shoulder = (person_kps[5][0].item() / frame_width, person_kps[5][1].item() / frame_height)
        right_shoulder = (person_kps[6][0].item() / frame_width, person_kps[6][1].item() / frame_height)
        left_wrist = (person_kps[9][0].item() / frame_width, person_kps[9][1].item() / frame_height)
        right_wrist = (person_kps[10][0].item() / frame_width, person_kps[10][1].item() / frame_height)

        if (person_kps[5][2] < 0.3 or person_kps[6][2] < 0.3 or
            person_kps[9][2] < 0.3 or person_kps[10][2] < 0.3):
            return None

        x_center = (left_shoulder[0] + right_shoulder[0]) / 2

        person = Person(
            x_center=x_center,
            left_wrist=left_wrist,
            right_wrist=right_wrist,
            left_shoulder=left_shoulder,
            right_shoulder=right_shoulder,
            person_id="DETECTED"
        )

        return person

    def run(self, camera_index: int = 0):
        """Run main loop"""
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print("Unable to open camera")
            print("Tip: Check camera permissions (System Settings > Privacy & Security > Camera)")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("Body Circuit Band - Solo Test Mode Started")
        print("Note: This is solo test version - your pose will be replicated to 3 positions")
        print("   - Raise both hands above shoulders to trigger circuit")
        print()
        print("Layered Music System (based on pose stability):")
        print("   - Very stable -> Layer 3 (Drums + Bass + Harmony)")
        print("   - Moderately stable -> Layer 2 (Drums + Bass)")
        print("   - Slight movement -> Layer 1 (Drums only)")
        print("   Tip: Raise hands and stay completely still to experience layered music!")
        print()
        print("Controls:")
        print("   - Press 'q': Exit program")
        print("   - Press 'ESC': Exit program")
        print("   - Close window: Exit program")
        print("\nDebug mode: Stability and circuit status will be printed to terminal\n")

        frame_count = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Camera read failed")
                    break

                frame = cv2.flip(frame, 1)
                frame_height, frame_width = frame.shape[:2]
                frame_count += 1

                results = self.model(frame, verbose=False)
                detected_person = self.extract_person_yolo(results, frame_width, frame_height)

                if detected_person:
                    if frame_count % 30 == 0:
                        import sys
                        print(f"Frame {frame_count}: Person detected, hands_raised={detected_person.is_hands_raised()}", flush=True)
                        sys.stdout.flush()

                    virtual_persons, stability = self.solo_mode.create_virtual_persons(detected_person)

                    circuit_closed, d_avg, distances = self.circuit_detector.update(
                        virtual_persons[0], virtual_persons[1], virtual_persons[2]
                    )

                    if frame_count % 30 == 0:
                        import sys
                        print(f"   Stability: {stability:.2f} (1.0=still, 0.0=unstable)", flush=True)
                        print(f"   Distance: A->B={distances[0]:.3f}, B->C={distances[1]:.3f}, C->A={distances[2]:.3f}", flush=True)
                        print(f"   Avg distance: {d_avg:.3f}, Circuit: {'closed' if circuit_closed else 'open'}", flush=True)
                        sys.stdout.flush()

                    if circuit_closed and not self.previous_circuit_state:
                        print("\nCircuit changed from open to closed - starting music!")
                        self.audio_controller.start_playback()
                    elif not circuit_closed and self.previous_circuit_state:
                        print("\nCircuit changed from closed to open - stopping music!")
                        self.audio_controller.stop_playback()

                    if circuit_closed:
                        self.audio_controller.set_volume_layered(d_avg, max_distance=0.30)

                    self.previous_circuit_state = circuit_closed

                    for person in virtual_persons:
                        self.visual.draw_person_landmarks(frame, person, frame_width, frame_height)

                    self.visual.draw_connections(
                        frame, virtual_persons[0], virtual_persons[1], virtual_persons[2],
                        distances, circuit_closed, frame_width, frame_height,
                        self.circuit_detector.distance_threshold
                    )

                    volume = max(0.0, min(1.0, 1.0 - (d_avg / 0.15))) if circuit_closed else 0.0

                    frame = self.visual.draw_status_panel(frame, circuit_closed, d_avg, volume, is_solo_mode=True)

                else:
                    cv2.putText(frame, "No person detected", (20, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    cv2.putText(frame, "Stand in front of camera", (20, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

                    if frame_count % 30 == 0:
                        print(f"Frame {frame_count}: No person detected, please stand in front of camera")

                help_text = "Press 'q' or ESC to quit"
                cv2.putText(frame, help_text, (20, frame_height - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow('Body Circuit Band - Solo Test', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print(f"\nExit key pressed, program ran {frame_count} frames")
                    break

                if cv2.getWindowProperty('Body Circuit Band - Solo Test', cv2.WND_PROP_VISIBLE) < 1:
                    print(f"\nWindow closed, program ran {frame_count} frames")
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

    band = BodyCircuitBandSolo(drum_path, bass_path, harmony_path)
    band.run(camera_index=0)


if __name__ == "__main__":
    main()
