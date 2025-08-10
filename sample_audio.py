# sample_audio.py
"""
Creates sample_audio.wav for the demo.
Preferred on macOS: uses 'say' + ffmpeg (no extra deps).
If 'say' is unavailable, falls back to pyttsx3 (may require pyobjc on macOS).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # set early

import shutil
import subprocess

TEXT = (
    "Hello team. We have two action items from today's meeting. "
    "First, we will show the demo to the customer on Wednesday. "
    "Second, we will prepare the pricing proposal by Friday. "
    "The customer found the price high, but they liked our support quality."
)

def make_with_say() -> bool:
    """Try to create sample_audio.wav using macOS 'say' and ffmpeg."""
    if shutil.which("say") and shutil.which("ffmpeg"):
        aiff = "sample_audio.aiff"
        subprocess.run(["say", TEXT, "-o", aiff], check=True)
        subprocess.run(["ffmpeg", "-y", "-i", aiff, "sample_audio.wav"], check=True)
        os.remove(aiff)
        print("sample_audio.wav created via macOS 'say'.")
        return True
    return False

def make_with_pyttsx3() -> bool:
    """Fallback to pyttsx3 if 'say' is not available."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        engine.save_to_file(TEXT, "sample_audio.wav")
        engine.runAndWait()
        print("sample_audio.wav created via pyttsx3.")
        return True
    except Exception as e:
        print("pyttsx3 fallback failed:", e)
        return False

if __name__ == "__main__":
    if make_with_say():
        pass
    elif make_with_pyttsx3():
        pass
    else:
        print("ERROR: Could not create sample audio. Install ffmpeg or pyttsx3 and try again.")
