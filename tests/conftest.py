import os

# Let tests create pygame windows without a real display.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
