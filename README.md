# Body Circuit Band 🎵

An interactive music installation that uses computer vision to detect people forming a circuit by holding hands. When the circuit is complete, music plays!

## 🌟 Features

- **Multi-person pose detection** using YOLOv8-Pose
- **Interactive circuit detection** - hold hands to complete the circuit
- **Dynamic audio playback** - volume adjusts based on hand proximity
- **Multiple modes**:
  - 2-person mode
  - 3-person mode
  - Interactive mode (supports any number of people)
  - Solo test mode (single person testing)

## 📋 Requirements

- Python 3.8+
- Webcam
- Audio files (drum, bass, harmony tracks)

## 🚀 Installation

1. Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/body_circuit_band.git
cd body_circuit_band
```

2. Install dependencies:
```bash
pip install opencv-python numpy pygame ultralytics
```

3. Make sure you have the audio files in `audio_samples_v2/` folder:
   - `drum.wav`
   - `bass.wav`
   - `harmony.wav`

## 🎮 Usage

### Interactive Mode (Recommended)
Choose the number of people dynamically:
```bash
python body_circuit_band_interactive.py
```

### 2-Person Mode
```bash
python body_circuit_band_two.py
```

### 3-Person Mode
```bash
python body_circuit_band_full.py
```

### Solo Test Mode
Test with just one person (simulates 3 people):
```bash
python body_circuit_band_solo_test.py
```

## 🎯 How It Works

1. **Stand in front of the camera** - the system detects all participants
2. **Form a circle and hold hands** - each person's right hand connects to the next person's left hand
3. **Complete the circuit** - when all hands are connected, music starts playing!
4. **Release hands** - music stops when the circuit breaks

## ⚙️ Configuration

Key parameters (in each file):
- `distance_threshold`: Maximum distance for hands to be considered "connected" (default: 0.3)
- `debounce_frames_close`: Frames needed to trigger music start (default: 8)
- `debounce_frames_open`: Frames needed to trigger music stop (default: 2)

## 🎨 Visual Feedback

- **Red/Green/Blue lines**: Connection status between people
- **Green**: Circuit closed (music playing)
- **Yellow**: Hands close but not quite connected
- **Red**: Hands too far apart

## 🛠️ Technical Details

- **Pose Detection**: YOLOv8n-pose model
- **Audio Engine**: Pygame mixer
- **Computer Vision**: OpenCV
- **Key Points Tracked**: Shoulders and wrists

## 📝 Controls

- Press `q` or `ESC` to quit
- Close window to exit

## 🤝 Contributing

Feel free to open issues or submit pull requests!

## 📄 License

This project is open source and available under the MIT License.

## 🎓 Course Project

This project was created for CMU course 48-727: Inquiry to Creative Design.

---

**Enjoy making music together!** 🎶
