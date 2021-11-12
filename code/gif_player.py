from matplotlib.widgets import Button
import matplotlib.pyplot as plt
import time


class GifPlayer:

    def __init__(self, mdp, title=""):

        # Create the figure and axes
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.07)
        self.ax.autoscale_view(True, True, True)


        self.images_frames = []
        self.current_index = 0
        self.mdp = mdp
        self.base_image = self.make_base_image(self.mdp.initialise_grid(grid_size=1.5)[2])
        self.current_frame = None

        # Show the env name in the window title
        self.fig.canvas.set_window_title(title)

        # Turn off x/y axis numbering/ticks
        self.ax.xaxis.set_ticks_position('none')
        self.ax.yaxis.set_ticks_position('none')
        _ = self.ax.set_xticklabels([])
        _ = self.ax.set_yticklabels([])

        # Flag indicating the window was closed
        self.closed = False

        def close_handler(evt):
            self.closed = True

        self.axreverse = self.fig.add_axes([0.3, 0.00, 0.1, 0.075])
        self.breverse = Button(self.axreverse, '\u00AB')
        self.breverse.on_clicked(self.reverse_button_handler)

        self.axprev = self.fig.add_axes([0.4, 0.00, 0.1, 0.075])
        self.bprev = Button(self.axprev, '\u2190')
        self.bprev.on_clicked(self.prev_button_handler)

        self.axnext = self.fig.add_axes([0.5, 0.00, 0.1, 0.075])
        self.bnext = Button(self.axnext, '\u2192')
        self.bnext.on_clicked(self.next_button_handler)

        self.axplay = self.fig.add_axes([0.6, 0.00, 0.1, 0.075])
        self.bplay = Button(self.axplay, '\u00BB')
        self.bplay.on_clicked(self.play_button_handler)

        self.fig.canvas.mpl_connect('key_press_event', self.key_handler)
        self.fig.canvas.mpl_connect('close_event', close_handler)
        self.fig.canvas.mpl_connect('Next', self.next_button_handler)

    def make_base_image(self, base_image):
        base_image = self.ax.imshow(base_image, origin="lower")
        return base_image.get_array()

    def add_value_function_frame(self, value_function):
        pass

    def add_q_function_frame(self, q_function):
        pass

    def add_stochastic_policy_frame(self, policy):
        image_frame = []
        for y in range(0, self.mdp.height):
            for x in range(0, self.mdp.width):
                prob_left = policy.get_probability((x, y), self.mdp.LEFT)
                prob_right = policy.get_probability((x, y), self.mdp.RIGHT)
                if self.mdp.height > 1:
                    prob_up = policy.get_probability((x, y), self.mdp.UP)
                    prob_down = policy.get_probability((x, y), self.mdp.DOWN)

                if (x, y) in self.mdp.goal_states:
                    text = self.ax.text(
                        x,
                        y,
                        f"{self.mdp.get_goal_states()[(x, y)]:+0.2f}",
                        fontsize="x-large",
                        horizontalalignment="center",
                        verticalalignment="center",
                    )
                elif (x, y) not in self.mdp.blocked_states:
                    if self.mdp.height > 1:
                        text = self.ax.text(
                            x,
                            y,
                            f"{prob_up:0.2f}\n{self.mdp.UP}\n{prob_left:0.2f}{self.mdp.LEFT} {self.mdp.RIGHT}{prob_right:0.2f}\n{self.mdp.DOWN}\n{prob_down:0.2f}",
                            fontsize="medium",
                            horizontalalignment="center",
                            verticalalignment="center",
                        )
                    else:
                        text = self.ax.text(
                            x,
                            y,
                            f"{prob_left:0.2f}{self.mdp.LEFT} {self.mdp.RIGHT}{prob_right:0.2f}",
                            fontsize="medium",
                            horizontalalignment="center",
                            verticalalignment="center",
                        )

                image_frame.append(text)
        self.images_frames.append(image_frame)

    def add_policy_frame(self, policy):
        image_frame = []
        arrow_map = {self.mdp.UP: '\u2191',
                     self.mdp.DOWN: '\u2193',
                     self.mdp.LEFT: '\u2190',
                     self.mdp.RIGHT: '\u2192',
                     }
        for y in range(self.mdp.height):
            for x in range(self.mdp.width):
                if (x, y) not in self.mdp.blocked_states and (x, y) not in self.mdp.goal_states:
                    if policy.select_action((x, y)) != self.mdp.TERMINATE:
                        action = arrow_map[policy.select_action((x, y))]
                        fontsize = "xx-large"
                    text = self.ax.text(
                        x,
                        y,
                        action,
                        fontsize=fontsize,
                        horizontalalignment="center",
                        verticalalignment="center",
                    )
                elif (x, y) in self.mdp.goal_states:
                    text = self.ax.text(
                        x,
                        y,
                        f"{self.mdp.get_goal_states()[(x, y)]:+0.2f}",
                        fontsize="x-large",
                        horizontalalignment="center",
                        verticalalignment="center",
                    )
                image_frame.append(text)
        self.images_frames.append(image_frame)

    def next_button_handler(self, event):
        self.current_index = min(len(self.images_frames) - 1, self.current_index + 1)
        self.render()

    def prev_button_handler(self, event):
        self.current_index = max(0, self.current_index - 1)
        self.render()

    def reverse_button_handler(self, event):
        while self.current_index > 1:
            self.current_index = max(0, self.current_index - 1)
            self.render()
            time.sleep(0.05)

    def play_button_handler(self, event):
        while self.current_index < len(self.images_frames) - 1:
            self.current_index = min(len(self.images_frames) - 1, self.current_index + 1)
            self.render()
            time.sleep(0.05)

    def key_handler(self, event):

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
            self.current_index = min(len(self.images_frames) - 1, self.current_index + 1)
            self.render()
            return

    def render(self):
        """
        Show an image or update the image being shown
        """
        if self.current_frame is not None:
            for cell in self.current_frame:
                cell.set_visible(False)

        self.current_frame = self.images_frames[self.current_index]
        for cell in self.current_frame:
            cell.set_visible(True)

        self.ax.set_title(f"frame number: {self.current_index + 1}")

        self.fig.canvas.draw()

        # Let matplotlib process UI events
        # This is needed for interactive mode to work properly
        plt.pause(0.001)

    def show(self, block=True):
        """
        Show the window, and start an event loop
        """
        [plt.close(f) for f in plt.get_fignums() if f != self.fig.number]
        for frame in self.images_frames:
            for cell in frame:
                cell.set_visible(False)

        # If not blocking, trigger interactive mode
        if not block:
            plt.ion()

        self.current_frame = self.images_frames[self.current_index]
        plt.show()

    def close(self):
        """
        Close the window
        """

        plt.close()
        self.closed = True
