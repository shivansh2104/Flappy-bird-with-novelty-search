import pygame
import neat
import time
import os
import random

BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("images","base.png")))
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("images","bg.png")))

class Base:
    
    VEL = 8
    WIDTH = BASE_IMG.get_width()

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        if self.x1 + self.WIDTH <0:
            self.x1 = self.x2 + self.WIDTH
        if self.x2 + self.WIDTH <0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(BASE_IMG, (self.x1,self.y))
        win.blit(BASE_IMG, (self.x2,self.y))
