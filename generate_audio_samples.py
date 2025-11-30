"""
Generate sample audio files
Use numpy and scipy to generate simple audio samples for testing
"""

import numpy as np
from scipy.io import wavfile
import os


def generate_drum_beat(duration=2.0, sample_rate=44100):
    """Generate drum beat sound effect"""
    t = np.linspace(0, duration, int(sample_rate * duration))

    drum = np.zeros_like(t)

    beat_interval = 0.5
    for beat_time in np.arange(0, duration, beat_interval):
        beat_start = int(beat_time * sample_rate)
        beat_length = int(0.1 * sample_rate)

        if beat_start + beat_length < len(t):
            beat_t = np.linspace(0, 0.1, beat_length)
            kick = np.sin(2 * np.pi * 80 * beat_t) * np.exp(-beat_t * 20)

            snare = np.random.randn(beat_length) * 0.3 * np.exp(-beat_t * 30)

            drum[beat_start:beat_start + beat_length] += kick + snare

    drum = drum / np.max(np.abs(drum)) * 0.7

    return (drum * 32767).astype(np.int16)


def generate_bass_line(duration=2.0, sample_rate=44100):
    """Generate bass line sound effect"""
    t = np.linspace(0, duration, int(sample_rate * duration))

    notes = [82.41, 98.00, 110.00, 73.42]
    note_duration = duration / len(notes)

    bass = np.zeros_like(t)

    for i, freq in enumerate(notes):
        start_idx = int(i * note_duration * sample_rate)
        end_idx = int((i + 1) * note_duration * sample_rate)

        if end_idx > len(t):
            end_idx = len(t)

        note_t = t[start_idx:end_idx] - t[start_idx]

        note = (np.sin(2 * np.pi * freq * note_t) * 0.6 +
                np.sin(2 * np.pi * freq * 2 * note_t) * 0.3 +
                np.sin(2 * np.pi * freq * 3 * note_t) * 0.1)

        envelope = np.exp(-note_t * 2)
        bass[start_idx:end_idx] = note * envelope

    bass = bass / np.max(np.abs(bass)) * 0.7

    return (bass * 32767).astype(np.int16)


def generate_harmony(duration=2.0, sample_rate=44100):
    """Generate harmony sound effect"""
    t = np.linspace(0, duration, int(sample_rate * duration))

    freq_e = 164.81
    freq_g = 196.00
    freq_b = 246.94

    harmony = (np.sin(2 * np.pi * freq_e * t) * 0.33 +
               np.sin(2 * np.pi * freq_g * t) * 0.33 +
               np.sin(2 * np.pi * freq_b * t) * 0.33)

    vibrato = 1 + 0.02 * np.sin(2 * np.pi * 5 * t)
    harmony = harmony * vibrato

    fade_length = int(0.1 * sample_rate)
    fade_in = np.linspace(0, 1, fade_length)
    fade_out = np.linspace(1, 0, fade_length)

    harmony[:fade_length] *= fade_in
    harmony[-fade_length:] *= fade_out

    harmony = harmony / np.max(np.abs(harmony)) * 0.6

    return (harmony * 32767).astype(np.int16)


def main():
    """Generate all audio samples"""
    output_dir = "audio_samples"
    os.makedirs(output_dir, exist_ok=True)

    sample_rate = 44100
    duration = 2.0

    print("Generating audio samples...")

    print("  - Generating drums (drum.wav)")
    drum = generate_drum_beat(duration, sample_rate)
    wavfile.write(os.path.join(output_dir, "drum.wav"), sample_rate, drum)

    print("  - Generating bass (bass.wav)")
    bass = generate_bass_line(duration, sample_rate)
    wavfile.write(os.path.join(output_dir, "bass.wav"), sample_rate, bass)

    print("  - Generating harmony (harmony.wav)")
    harmony = generate_harmony(duration, sample_rate)
    wavfile.write(os.path.join(output_dir, "harmony.wav"), sample_rate, harmony)

    print("Audio samples generated successfully!")
    print(f"   Files saved to: {output_dir}/")


if __name__ == "__main__":
    main()
