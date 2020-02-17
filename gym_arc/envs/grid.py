'''
ARC - Grid Representation Object
Author: Yuya Jeremy Ong (yuyajeremyong@gmail.com)
'''
from __future__ import print_function
import cv2
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class Grid:
    def __init__(self, width=3, height=3, grid=None):
        # Global Constants
        self.cmap = colors.ListedColormap(['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
                                      '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
        self.norm = colors.Normalize(vmin=0, vmax=9)

        # Check for existing grid implementation
        if grid is None:
            self.state = np.zeros((height, width), dtype=int)
        else:
            self.set_grid(grid)

        # Set Width and Height Property
        self.height, self.width = self.state.shape

    def get(self, x, y):
        assert(self.check_bounds(x, y))
        return self.state[x][y]

    def set(self, x, y, symbol):
        assert(self.check_bounds(x, y))
        if symbol < 0 and symbol >= len(self.cmap):
            raise ValueError('Invalid Symbol')
        self.state[x][y] = symbol

    def set_grid(self, grid):
        # Check for existing grid implementation
        if type(grid) == list or isinstance(grid, np.ndarray):
            self.state = np.array(grid).copy()
        elif type(grid) == Grid:
            self.state = grid.state.copy()

        # Update Width and Height Property
        self.height, self.width = self.state.shape

    def reset_grid(self):
        self.state = np.zeros((self.height, self.width), dtype=int)

    def resize_grid(self, x, y):
        if x < self.height:
            self.state = self.state[:x, :]
        elif x >= self.height:
            self.state = np.append(self.state, np.zeros((x-self.height, self.width)), axis=0)

        # Update Width and Height Property
        self.height, self.width = self.state.shape

        if y < self.width:
            self.state = self.state[:, :y]
        elif y >= self.width:
            self.state = np.append(self.state, np.zeros((self.height, y-self.width)), axis=1)

        # Update Width and Height Property
        self.height, self.width = self.state.shape

    # Recursive Function Call for Flooding
    # TODO: Fix recursion inf depth case
    def flood_fill(self, i, j, symbol, target):
        if self.check_bounds(i, j):
            if self.state[i][j] == target:
                self.state[i][j] = symbol
                self.flood_fill(i-1, j, symbol, target)
                self.flood_fill(i+1, j, symbol, target)
                self.flood_fill(i, j-1, symbol, target)
                self.flood_fill(i, j+1, symbol, target)

    # Wrapper Function for Flood
    def flood_grid(self, x, y, symbol):
        # Flood the Grid
        self.flood_fill(x, y, symbol, self.state[x][y])

    def copy_paste_grid(self, input_grid, x, y, i, j, x_target, y_target):
        # Check Bounds
        assert(input_grid.check_bounds(x, y))
        assert(input_grid.check_bounds(x+i, y+j))
        assert(self.check_bounds(x_target, y_target))
        assert(x+i >= x and y+j >= j)

        # Handle Boundary Cases
        if (self.height - x_target - 1) < i:
            x_max = self.height
            x_i = self.height - x_target
        else:
            x_max = x_target + i
            x_i = i

        if (self.width - y_target - 1) < j:
            y_max = self.width
            y_j = self.width - y_target
        else:
            y_max = y_target + j
            y_j = j

        self.state[x_target:x_max, y_target:y_max] = input_grid.state[x:x+x_i, x:y+y_j]

    def check_bounds(self, i, j):
        return (i >= 0 and i < self.height) and (j >= 0 and j < self.width)

    def plot(self, title=''):
        fig, ax = plt.subplots()
        ax.imshow(self.state, cmap=self.cmap, norm=self.norm)
        ax.grid(True,which='both',color='lightgrey', linewidth=0.5)
        ax.set_yticks([x-0.5 for x in range(1+len(self.state))])
        ax.set_xticks([x-0.5 for x in range(1+len(self.state[0]))])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(title)
        plt.show()

    def get_plot(self, title=''):
        fig, ax = plt.subplots()
        ax.imshow(self.state, cmap=self.cmap, norm=self.norm)
        ax.grid(True,which='both',color='lightgrey', linewidth=0.5)
        ax.set_yticks([x-0.5 for x in range(1+len(self.state))])
        ax.set_xticks([x-0.5 for x in range(1+len(self.state[0]))])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(title)

        return fig

    def fig2img(self, fig, img_height):
        canvas = FigureCanvas(fig)
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        out = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        del canvas # Free Up Memory
        plt.close()
        return out

    def render_env(self, desired_size=256):
        fig = self.get_plot()
        im = self.fig2img(fig, desired_size)
        old_size = im.shape[:2] # old_size is in (height, width) format
        plt.close()

        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        # new_size should be in (width, height) format
        im = cv2.resize(im, (new_size[1], new_size[0]))

        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        color = [255, 255, 255]
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        new_im = cv2.cvtColor(new_im, cv2.COLOR_BGR2RGB)

        return new_im