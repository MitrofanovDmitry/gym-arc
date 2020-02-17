'''
ARC Base Environment for OpenAI Gym
Author: Yuya J. Ong (yuyajeremyong@gmail.com)
'''
from __future__ import print_function
import os
import cv2
import time
import signal
import logging
import subprocess

import gym
from gym.utils import seeding
from gym import error, spaces, utils

# from gym_arc.envs.rendering import SimpleImageViewer
from grid import Grid
from rendering import SimpleImageViewer

logger = logging.getLogger(__name__)

class GridEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, width=3, height=3, img_size=512):
        self.grid = Grid(width, height)
        self.viewer = SimpleImageViewer()
        self.img_size = img_size

    def step(self, action):
        # Parse Command
        act = action.split(' ')

        # Apply Command
        if act[0] == 'GRID':
            self.grid = Grid(int(act[1]), int(act[2]))
        elif act[0] == 'EDIT':
            self.grid.set(int(act[1]), int(act[2]), int(act[3]))
        elif act[0] == 'FILL':
            self.grid.flood_grid(int(act[1]), int(act[2]), int(act[3]))
        # elif act[0] == 'SELECT_FILL':
        #     self.grid.set(int(act[1]), int(act[2]), int(act[3]))
        elif act[0] == 'RESIZE':
            self.grid.resize_grid(int(act[1]), int(act[2]))
        elif act[0] == 'RESET':
            self.grid.reset_grid()

    def reset(self):
        self.grid = Grid(3, 3)

    def render(self, mode='human'):
        self.viewer.imshow(self.grid.render_env(self.img_size))