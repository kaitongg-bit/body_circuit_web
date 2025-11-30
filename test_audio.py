"""
Test audio playback functionality
"""
import pygame
import time

print("Testing audio system...")

pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
print("Pygame mixer initialized successfully")

try:
    drum = pygame.mixer.Sound("audio_samples/drum.wav")
    bass = pygame.mixer.Sound("audio_samples/bass.wav")
    harmony = pygame.mixer.Sound("audio_samples/harmony.wav")
    print("Audio files loaded successfully")
    print(f"   - drum.wav duration: {drum.get_length():.2f}s")
    print(f"   - bass.wav duration: {bass.get_length():.2f}s")
    print(f"   - harmony.wav duration: {harmony.get_length():.2f}s")
except Exception as e:
    print(f"Audio file loading failed: {e}")
    exit(1)

print("\nStarting playback test (5 seconds)...")
print("   If you hear three different tones mixed, the audio system is working")

try:
    ch1 = drum.play(loops=-1)
    ch2 = bass.play(loops=-1)
    ch3 = harmony.play(loops=-1)

    print("Playback started")

    time.sleep(5)

    drum.stop()
    bass.stop()
    harmony.stop()

    print("Playback test completed")

except Exception as e:
    print(f"Playback failed: {e}")

finally:
    pygame.mixer.quit()
    print("\nTest ended")
