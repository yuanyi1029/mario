import ctypes

import pyglet
from pyglet import gl

ASPECT_RATIO = 4 / 3

class FrameViewer:

    def __init__(self):
        self._window = None
        self.is_open = True

    def _create_window(self, frame):
        assert (
            len(frame.shape) == 3 and frame.shape[2] == 3
        ), "must be an RGB image"

        image_height, image_width = frame.shape[:2]

        # guess a screen size that doesn't distort the image too much but also is not tiny or huge
        display = pyglet.canvas.get_display()
        screen = display.get_default_screen()
        max_win_width = screen.width * 0.9
        max_win_height = screen.height * 0.9
        win_width = image_width
        win_height = int(win_width / ASPECT_RATIO)

        while win_width > max_win_width or win_height > max_win_height:
            win_width //= 2
            win_height //= 2
        while win_width < max_win_width / 2 and win_height < max_win_height / 2:
            win_width *= 2
            win_height *= 2

        self._window = pyglet.window.Window(width=win_width, height=win_height)
        self._window.on_close = self._on_close

        gl.glEnable(gl.GL_TEXTURE_2D)
        self._texture_id = gl.GLuint(0)
        gl.glGenTextures(1, ctypes.byref(self._texture_id))
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA8,
            image_width,
            image_height,
            0,
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            None,
        )

    def _draw(self, frame):
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture_id)
        video_buffer = ctypes.cast(
            frame.tobytes(),
            ctypes.POINTER(ctypes.c_short),
        )
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D,
            0,
            0,
            0,
            frame.shape[1],
            frame.shape[0],
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            video_buffer,
        )

        x = 0
        y = 0
        w = self._window.width
        h = self._window.height

        pyglet.graphics.draw(
            4,
            pyglet.gl.GL_QUADS,
            ("v2f", [x, y, x + w, y, x + w, y + h, x, y + h]),
            ("t2f", [0, 1, 1, 1, 1, 0, 0, 0]),
        )

    def _on_close(self):
        self.is_open = False
        if self._window is not None:
            self._window.set_visible(False)

    def show(self, frame):
        if not self.is_open:
            return

        if self._window is None:
            self._create_window(frame)

        self._window.switch_to()
        self._window.dispatch_events()
        if not self.is_open:
            return

        self._draw(frame)
        self._window.flip()