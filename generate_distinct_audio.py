"""
Generate audio samples with high contrast
Make differences between layers more noticeable
"""

import numpy as np
from scipy.io import wavfile
import os


def generate_powerful_drum(duration=2.0, sample_rate=44100):
    """Generate powerful drum beat (Layer 1 foundation)"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    drum = np.zeros_like(t)

    beat_interval = 0.5
    for beat_time in np.arange(0, duration, beat_interval):
        beat_start = int(beat_time * sample_rate)
        beat_length = int(0.15 * sample_rate)

        if beat_start + beat_length < len(t):
            beat_t = np.linspace(0, 0.15, beat_length)

            kick = np.sin(2 * np.pi * 60 * beat_t) * np.exp(-beat_t * 15)

            snare_time = beat_time + 0.25
            if snare_time < duration:
                snare_start = int(snare_time * sample_rate)
                snare_length = int(0.08 * sample_rate)
                if snare_start + snare_length < len(t):
                    snare_t = np.linspace(0, 0.08, snare_length)
                    snare = np.random.randn(snare_length) * 0.5 * np.exp(-snare_t * 25)
                    drum[snare_start:snare_start + snare_length] += snare

            drum[beat_start:beat_start + beat_length] += kick * 1.2

    drum = drum / np.max(np.abs(drum)) * 0.9
    return (drum * 32767).astype(np.int16)


def generate_groovy_bass(duration=2.0, sample_rate=44100):
    """Generate groovy bass line (Layer 2 enhancement)"""
    t = np.linspace(0, duration, int(sample_rate * duration))

    notes = [
        (0.00, 0.25, 82.41),
        (0.25, 0.50, 110.00),
        (0.50, 0.75, 98.00),
        (0.75, 1.25, 73.42),
        (1.25, 1.50, 82.41),
        (1.50, 2.00, 110.00),
    ]

    bass = np.zeros_like(t)

    for start_time, end_time, freq in notes:
        start_idx = int(start_time * sample_rate)
        end_idx = int(end_time * sample_rate)

        if end_idx > len(t):
            end_idx = len(t)

        note_t = t[start_idx:end_idx] - t[start_idx]

        note = (np.sin(2 * np.pi * freq * note_t) * 0.5 +
                np.sin(2 * np.pi * freq * 2 * note_t) * 0.25 +
                np.sin(2 * np.pi * freq * 3 * note_t) * 0.15 +
                np.sin(2 * np.pi * (freq - 5) * note_t) * 0.1)

        envelope = np.exp(-note_t * 3)
        bass[start_idx:end_idx] += note * envelope

    bass = bass / np.max(np.abs(bass)) * 0.85
    return (bass * 32767).astype(np.int16)


def generate_bright_harmony(duration=2.0, sample_rate=44100):
    """Generate bright harmony (Layer 3 embellishment)"""
    t = np.linspace(0, duration, int(sample_rate * duration))

    harmony = np.zeros_like(t)

    chords = [
        (0.0, 1.0, [329.63, 392.00, 493.88]),
        (1.0, 2.0, [440.00, 523.25, 659.25]),
    ]

    for start_time, end_time, freqs in chords:
        start_idx = int(start_time * sample_rate)
        end_idx = int(end_time * sample_rate)

        if end_idx > len(t):
            end_idx = len(t)

        chord_t = t[start_idx:end_idx] - t[start_idx]

        chord = sum(np.sin(2 * np.pi * f * chord_t) for f in freqs) / len(freqs)

        vibrato = 1 + 0.03 * np.sin(2 * np.pi * 5 * chord_t)
        chord = chord * vibrato

        harmony[start_idx:end_idx] += chord

    fade_length = int(0.05 * sample_rate)
    fade_in = np.linspace(0, 1, fade_length)
    fade_out = np.linspace(1, 0, fade_length)

    harmony[:fade_length] *= fade_in
    harmony[-fade_length:] *= fade_out

    harmony = harmony / np.max(np.abs(harmony)) * 0.75
    return (harmony * 32767).astype(np.int16)


def main():
    """Generate audio samples with high contrast"""
    output_dir = "audio_samples_v2"
    os.makedirs(output_dir, exist_ok=True)

    sample_rate = 44100
    duration = 2.0

    print("Generating high contrast audio samples...")
    print("   Layer contrast:")
    print("   Layer 1: Powerful kick drum (60Hz)")
    print("   Layer 2: + Groovy bass (73-110Hz)")
    print("   Layer 3: + Bright harmony (330-660Hz)")
    print()

    print("  - Generating powerful drums (drum.wav)")
    drum = generate_powerful_drum(duration, sample_rate)
    wavfile.write(os.path.join(output_dir, "drum.wav"), sample_rate, drum)

    print("  - Generating groovy bass (bass.wav)")
    bass = generate_groovy_bass(duration, sample_rate)
    wavfile.write(os.path.join(output_dir, "bass.wav"), sample_rate, bass)

    print("  - Generating bright harmony (harmony.wav)")
    harmony = generate_bright_harmony(duration, sample_rate)
    wavfile.write(os.path.join(output_dir, "harmony.wav"), sample_rate, harmony)

    print()
    print("New audio samples generated successfully!")
    print(f"   Saved to: {output_dir}/")
    print()
    print("Improvements:")
    print("   1. Lower frequency, more powerful drums (60Hz vs 80Hz)")
    print("   2. Bass with enhanced groove and harmonic richness")
    print("   3. Harmony raised one octave, brighter and easier to distinguish (330-660Hz)")
    print("   4. Three layers with completely separated frequency bands")


if __name__ == "__main__":
    main()
