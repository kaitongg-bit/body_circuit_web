"""
Test layered audio playback
Manually switch between different layers to clearly feel the difference
"""
import pygame
import time

print("Layered Audio Test Program")
print("=" * 50)

pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

drum = pygame.mixer.Sound("audio_samples_v2/drum.wav")
bass = pygame.mixer.Sound("audio_samples_v2/bass.wav")
harmony = pygame.mixer.Sound("audio_samples_v2/harmony.wav")

print("Audio files loaded successfully")
print(f"   - drum.wav: {drum.get_length():.2f}s")
print(f"   - bass.wav: {bass.get_length():.2f}s")
print(f"   - harmony.wav: {harmony.get_length():.2f}s")
print()

ch_drum = drum.play(loops=-1)
ch_bass = bass.play(loops=-1)
ch_harmony = harmony.play(loops=-1)

ch_drum.set_volume(0)
ch_bass.set_volume(0)
ch_harmony.set_volume(0)

print("Control instructions:")
print("   Will play 3 layers sequentially, 5 seconds each")
print("   Listen carefully to the music changes!")
print("=" * 50)
print()

try:
    print("Layer 1 - Drums only")
    ch_drum.set_volume(0.8)
    ch_bass.set_volume(0.0)
    ch_harmony.set_volume(0.0)
    time.sleep(5)

    print("Layer 2 - Drums + Bass (notice the bass joining!)")
    ch_drum.set_volume(0.8)
    ch_bass.set_volume(0.8)
    ch_harmony.set_volume(0.0)
    time.sleep(5)

    print("Layer 3 - Drums + Bass + Harmony (full band!)")
    ch_drum.set_volume(0.8)
    ch_bass.set_volume(0.8)
    ch_harmony.set_volume(0.8)
    time.sleep(5)

    print()
    print("Now reversing...")
    print()

    print("Back to Layer 2")
    ch_harmony.set_volume(0.0)
    time.sleep(5)

    print("Back to Layer 1")
    ch_bass.set_volume(0.0)
    time.sleep(5)

    print()
    print("Test complete!")
    print("Can you hear the difference?")
    print("If the difference is not obvious, you may need to replace the audio files.")

finally:
    drum.stop()
    bass.stop()
    harmony.stop()
    pygame.mixer.quit()
    print("\nProgram ended")
