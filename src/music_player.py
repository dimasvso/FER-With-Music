import os
import random
import pygame
from config import BASE_DIR

MUSIC_DIR = os.path.join(BASE_DIR, "music")


class MusicPlayer:
    def __init__(self):
        pygame.mixer.init()
        self.current_emotion = None

    def play(self, emotion):
        # Skip kalau emosi sama dan lagu masih jalan
        if emotion == self.current_emotion and pygame.mixer.music.get_busy():
            return

        folder = os.path.join(MUSIC_DIR, emotion)
        if not os.path.exists(folder):
            print(f"[MusicPlayer] Folder '{emotion}' tidak ditemukan")
            return

        songs = [f for f in os.listdir(folder) if f.endswith(".mp3")]
        if not songs:
            print(f"[MusicPlayer] Ga ada lagu di folder '{emotion}'")
            return

        chosen = random.choice(songs)
        song_path = os.path.join(folder, chosen)

        pygame.mixer.music.stop()
        pygame.mixer.music.load(song_path)
        pygame.mixer.music.play(-1)  # loop
        self.current_emotion = emotion
        print(f"Playing [{emotion}]: {chosen}")

    def stop(self):
        pygame.mixer.music.stop()
        self.current_emotion = None