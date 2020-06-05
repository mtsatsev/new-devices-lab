import cv2
import time
import pygame

class Volume(object):
    def __init__(self):
        self.level = .5

    def increase(self,amount):
        self.level += amount
    def decrease(self,amount):
        self.level -= amount



class Layout(baseLayout):
    def __init__(self,bgSubThreshold):
        self.bgSubThreshold = bgSubThreshold
        self.window = self.createLayout()

    def createLayout(self):
        pygame.init()
        screen = pygame.display.set_mode((800,800))
        pygame.display.set_caption("just for testing")
        return screen
