import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.backend_bases import RendererBase

class GifPlayer:

    def __init__(self, title=""):
        # create a list of images to show in the gif
        self.gif_fig = None
        self.imshow_obj = None
        self.images = []
        self.current_index = 0

        # Create the figure and axes
        self.gif_fig, self.gif_ax = plt.subplots()

        # Show the env name in the window title
        self.gif_fig.canvas.set_window_title(title)

        # Turn off x/y axis numbering/ticks
        self.gif_ax.xaxis.set_ticks_position('none')
        self.gif_ax.yaxis.set_ticks_position('none')
        _ = self.gif_ax.set_xticklabels([])
        _ = self.gif_ax.set_yticklabels([])

        # Flag indicating the window was closed
        self.closed = False

        def close_handler(evt):
            self.closed = True

        self.gif_fig.canvas.mpl_connect('key_press_event', self.key_handler)
        self.gif_fig.canvas.mpl_connect('close_event', close_handler)

    def add_image(self, image):
        # image = self.gif_ax.imshow(image)
        self.images.append(image.get_array())

    def key_handler(self, event):
        print('pressed', event.key)

        if event.key == 'escape':
            self.close()
            return

        if event.key == 'backspace':
            print("reset")
            return

        if event.key == 'left':
            self.current_index = max(0, self.current_index - 1)
            self.render()
            return

        if event.key == 'right':
            self.current_index = min(len(self.images) - 1, self.current_index + 1)
            self.render()
            return

    def render(self):
        """
        Show an image or update the image being shown
        """
        current_image = self.images[self.current_index]

        if self.imshow_obj is None:
            self.imshow_obj = self.gif_ax.imshow(current_image)

        # self.imshow_obj.set_data(current_image)
        self.gif_ax.set_title(f"frame number: {self.current_index + 1}")
        self.gif_ax.add_artist(current_image)
        self.gif_fig.canvas.draw()

        # Let matplotlib process UI events
        # This is needed for interactive mode to work properly
        plt.pause(0.001)

    def show(self, block=True):
        """
        Show the window, and start an event loop
        """
        [plt.close(f) for f in plt.get_fignums() if f != self.gif_fig.number]

        # def f(x, y):
        #     return np.sin(x) + np.cos(y)
        #
        # x = np.linspace(0, 2 * np.pi, 120)
        # y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

        # ims is a list of lists, each row is a list of artists to draw in the
        # current frame; here we are just animating one artist, the image, in
        # each frame
        # ims = []
        # for i in range(60):
        #     x += np.pi / 15.
        #     y += np.pi / 20.
        #     im = self.gif_ax.imshow(f(x, y), animated=True)
        #     if i == 0:
        #         self.gif_ax.imshow(f(x, y))  # show an initial one first
        #     ims.append([im])

        self.gif_ax.imshow(self.images[2])
        # ani = animation.ArtistAnimation(self.gif_fig, self.images, interval=50, blit=True,
        #                                 repeat_delay=1000)

        # If not blocking, trigger interactive mode
        if not block:
            plt.ion()

        # Show the plot
        # In non-interative mode, this enters the matplotlib event loop
        # In interactive mode, this call does not block
        plt.show()

    def close(self):
        """
        Close the window
        """

        plt.close()
        self.closed = True
