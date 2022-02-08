"""GUI application for thermal image analysis."""
import os
import pickle
import sys
from tkinter.messagebox import showwarning

import numpy as np
import pygame
import pygame_gui as pygui
from scipy.ndimage.interpolation import zoom
from thermal_base import ThermalImage
from thermal_base import utils as ThermalImageHelpers

from utils import WindowHandler, openImage, saveImage

pygame.init()
WINDOW_SIZE = (1020, 590)
NEW_FILE = False


class Manager(pygui.UIManager):
    """Class for manager.

    A manager is a set of menu buttons and descriptions.
    A manager is assigned to each page of the application.
    """

    def __init__(self, buttons, textbox=None, fields=None):
        """Initilizer for manager."""
        super().__init__(WINDOW_SIZE)
        self.buttons = [
            (
                pygui.elements.UIButton(
                    relative_rect=pygame.Rect(pos, size), text=text, manager=self
                ),
                func,
            )
            for pos, size, text, func in buttons
        ]
        if textbox:
            self.textbox = pygui.elements.ui_text_box.UITextBox(
                html_text=textbox[2],
                relative_rect=pygame.Rect(textbox[:2]),
                manager=self,
            )
        if fields:
            self.fields = [
                (
                    pygui.elements.ui_text_entry_line.UITextEntryLine(
                        relative_rect=pygame.Rect((pos[0], pos[1] + 40), size),
                        manager=self,
                    ),
                    pygui.elements.ui_text_box.UITextBox(
                        html_text=text,
                        relative_rect=pygame.Rect(pos, (-1, -1)),
                        manager=self,
                    ),
                )
                for pos, size, text in fields
            ]

    def process_events(self, event):
        """Process button presses."""
        if event.type == pygame.USEREVENT:
            if event.user_type == pygui.UI_BUTTON_PRESSED:
                for button, func in self.buttons:
                    if event.ui_element == button:
                        func()

        super().process_events(event)


class Window:
    """Class that handles the main window."""

    fonts = [
        pygame.font.SysFont("monospace", 20),
        pygame.font.SysFont("monospace", 24),
        pygame.font.SysFont("arial", 18),
    ]

    cursors = [
        pygame.image.load("./assets/cursors/pointer.png"),
        pygame.image.load("./assets/cursors/crosshair.png"),
    ]
    logo = pygame.transform.scale(pygame.image.load("./assets/DTlogo.png"), (100, 100))
    clip = lambda x, a, b: a if x < a else b if x > b else x

    @staticmethod
    def renderText(surface, text, location):
        """Render text at a given location."""
        whitetext = Window.fonts[2].render(text, 1, (255, 255, 255))
        Window.fonts[0].set_bold(True)
        blacktext = Window.fonts[2].render(text, 1, (0, 0, 0))
        Window.fonts[0].set_bold(False)

        textrect = whitetext.get_rect()
        for i in range(-3, 4):
            for j in range(-3, 4):
                textrect.center = [a + b for a, b in zip(location, (i, j))]
                surface.blit(blacktext, textrect)
        textrect.center = location
        surface.blit(whitetext, textrect)

    def __init__(self, thermal_image=None, filename=None):
        """Initializer for the main window."""
        self.exthandler = WindowHandler()
        if thermal_image is not None:
            mat = thermal_image.thermal_np.astype(np.float32)

            if mat.shape != (512, 640):
                y0, x0 = mat.shape
                mat = zoom(mat, [512 / y0, 640 / x0])

            self.mat = mat
            self.mat_orig = mat.copy()
            self.mat_emm = mat.copy()
            self.raw = thermal_image.raw_sensor_np
            self.meta = thermal_image.meta
            self.overlays = pygame.Surface((640, 512), pygame.SRCALPHA)
            self.thermalData = thermal_image.thermal_np + 273
        else:
            with open(filename, "rb") as f:
                data = pickle.load(f)
            self.mat = data.mat
            self.mat_orig = data.mat_orig
            self.mat_emm = data.mat_emm
            self.raw = data.raw
            self.meta = data.meta
            self.overlays = pygame.image.fromstring(data.overlays, (640, 512), "RGBA")

            for entry in data.tableEntries:
                self.exthandler.addToTable(entry)
            self.exthandler.loadGraph(data.plots)
            self.exthandler.addRects(data.rects)

        self.colorMap = "jet"
        self.lineNum = 0
        self.boxNum = 0
        self.spotNum = 0
        self.areaMode = "poly"
        self.selectionComplete = False
        self.work("colorMap", self.colorMap)

        self.mode = "main"
        # Dictionary of pages. Each page is a manager.
        self.managers = {}
        self.managers["main"] = Manager(
            buttons=[
                ((15, 15), (215, 45), "Spot marking", lambda: self.changeMode("spot")),
                (
                    (15, 75),
                    (215, 45),
                    "Line measurement",
                    lambda: self.changeMode("line"),
                ),
                ((15, 135), (215, 45), "Area marking", lambda: self.changeMode("area")),
                ((15, 195), (215, 45), "ROI scaling", lambda: self.changeMode("scale")),
                (
                    (15, 255),
                    (215, 45),
                    "Change colorMap",
                    lambda: self.changeMode("colorMap"),
                ),
                (
                    (15, 315),
                    (215, 45),
                    "Emissivity scaling",
                    lambda: self.changeMode("emissivity"),
                ),
                (
                    (15, 470),
                    (215, 45),
                    "Reset modifications",
                    lambda: self.work("reset"),
                ),
                ((15, 530), (100, 45), "Open", lambda: self.work("open")),
                ((130, 530), (100, 45), "Save", lambda: saveImage(self)),
            ]
        )
        self.managers["spot"] = Manager(
            buttons=[((15, 530), (215, 45), "Back", lambda: self.changeMode("main"))],
            textbox=((15, 15), (215, -1), "Click to mark spots"),
        )
        self.managers["line"] = Manager(
            buttons=[
                (
                    (15, 410),
                    (215, 45),
                    "Continue",
                    lambda: self.work("line") if len(self.linePoints) == 2 else None,
                ),
                ((15, 470), (215, 45), "Reset", lambda: self.changeMode("line")),
                ((15, 530), (215, 45), "Back", lambda: self.changeMode("main")),
            ],
            textbox=(
                (15, 15),
                (215, -1),
                "Click to mark the end points of the line. Click continue to get plot and reset to remove the line",
            ),
        )
        self.managers["area"] = Manager(
            buttons=[
                ((15, 470), (215, 45), "Continue", lambda: self.work("area")),
                ((15, 530), (215, 45), "Back", lambda: self.changeMode("main")),
            ],
            textbox=(
                (15, 15),
                (215, -1),
                "Click and drag to draw selection. Select continue to mark",
            ),
        )
        self.managers["scale"] = Manager(
            buttons=[
                (
                    (15, 270),
                    (215, 45),
                    "Switch to rect mode",
                    lambda: self.work("scale", "switchMode"),
                ),
                (
                    (15, 350),
                    (215, 45),
                    "Continue",
                    lambda: self.work("scale", "scale")
                    if self.selectionComplete
                    else None,
                ),
                (
                    (15, 410),
                    (215, 45),
                    "Reset scaling",
                    lambda: self.work("scale", "reset"),
                ),
                (
                    (15, 470),
                    (215, 45),
                    "Reset selection",
                    lambda: self.changeMode("scale"),
                ),
                ((15, 530), (215, 45), "Back", lambda: self.changeMode("main")),
            ],
            textbox=(
                (15, 15),
                (215, -1),
                "Click to mark vertices. Press Ctrl and click to close the selection",
            ),
        )
        self.managers["colorMap"] = Manager(
            buttons=[
                ((15, 15), (215, 45), "Jet", lambda: self.work("colorMap", "jet")),
                ((15, 75), (215, 45), "Hot", lambda: self.work("colorMap", "hot")),
                ((15, 135), (215, 45), "Cool", lambda: self.work("colorMap", "cool")),
                ((15, 195), (215, 45), "Gray", lambda: self.work("colorMap", "gray")),
                (
                    (15, 255),
                    (215, 45),
                    "Inferno",
                    lambda: self.work("colorMap", "inferno"),
                ),
                (
                    (15, 315),
                    (215, 45),
                    "Copper",
                    lambda: self.work("colorMap", "copper"),
                ),
                (
                    (15, 375),
                    (215, 45),
                    "Winter",
                    lambda: self.work("colorMap", "winter"),
                ),
                ((15, 530), (215, 45), "Back", lambda: self.changeMode("main")),
            ]
        )
        self.managers["emissivity"] = Manager(
            buttons=[
                (
                    (15, 410),
                    (215, 45),
                    "Continue",
                    lambda: self.work("emissivity", "update")
                    if self.selectionComplete
                    else None,
                ),
                (
                    (15, 470),
                    (215, 45),
                    "Reset",
                    lambda: self.work("emissivity", "reset"),
                ),
                ((15, 530), (215, 45), "Back", lambda: self.changeMode("main")),
            ],
            textbox=(
                (15, 15),
                (215, -1),
                "Select region, enter values and press continue. Click to mark vertices."
                "Press Ctrl and click to close the selection",
            ),
            fields=[
                ((15, 165), (215, 45), "Emissivity:"),
                ((15, 240), (215, 45), "Reflected Temp.:"),
                ((15, 315), (215, 45), "Atmospheric Temp.:"),
            ],
        )

        self.linePoints = []

        self.cursor_rect = self.cursors[0].get_rect()
        self.background = pygame.Surface(WINDOW_SIZE)
        self.background.fill((0, 0, 0))

    def changeMode(self, mode):
        """Change mode."""
        # Mode change - reset handler
        if self.mode == "line":
            if mode in ("main", "line"):
                self.linePoints = []

        if self.mode in ("scale", "area", "emissivity"):
            if mode in ("main", "scale", "area"):
                self.selectionComplete = False
                self.linePoints = []

        self.mode = mode

    def work(self, mode, *args):
        """Work based on mode."""
        if mode == "reset":
            # Resetting overlays and plots
            self.overlays = pygame.Surface((WINDOW_SIZE[0] - 245, 512), pygame.SRCALPHA)
            self.lineNum = 0
            self.boxNum = 0
            self.spotNum = 0
            self.exthandler.killThreads()
            self.exthandler = WindowHandler()

            # Resetting ROI scaling
            self.work("scale", "reset")
            # Resetting Emissivity changes
            self.work("emissivity", "reset")

        if mode == "open":
            global NEW_FILE
            NEW_FILE = True

        if mode == "line":
            self.lineNum += 1
            linePoints = [
                [a - b for a, b in zip(points, (245, 15))] for points in self.linePoints
            ]
            pygame.draw.line(
                self.overlays, (255, 255, 255), linePoints[0], linePoints[1], 3
            )

            center = (
                (linePoints[0][0] + linePoints[1][0]) / 2,
                (linePoints[0][1] + linePoints[1][1]) / 2,
            )

            self.renderText(self.overlays, f"l{self.lineNum}", center)

            self.exthandler.linePlot(
                self.mat_emm,
                f"l{self.lineNum}",
                np.array(linePoints[0][::-1]),
                np.array(linePoints[1][::-1]),
            )
            self.linePoints = []

        if mode == "spot":
            self.spotNum += 1
            self.renderText(
                self.overlays,
                f"s{self.spotNum}",
                (self.mx - 245 + 15, self.my - 15 - 13),
            )
            pygame.draw.line(
                self.overlays,
                (255, 255, 255),
                (self.mx - 245, self.my - 15 - 5),
                (self.mx - 245, self.my - 15 + 5),
                3,
            )
            pygame.draw.line(
                self.overlays,
                (255, 255, 255),
                (self.mx - 245 - 5, self.my - 15),
                (self.mx - 245 + 5, self.my - 15),
                3,
            )
            val = self.mat_emm[self.cy - 15, self.cx - 245]
            self.exthandler.addToTable([f"s{self.spotNum}", val, val, val])

        if mode == "area":
            if self.selectionComplete:
                points = [(a - 245, b - 15) for a, b in self.linePoints]
                x_coords, y_coords = zip(*points)
                xmin = min(x_coords)
                xmax = max(x_coords)
                ymin = min(y_coords)
                ymax = max(y_coords)
                if xmin == xmax or ymin == ymax:
                    return
                self.boxNum += 1
                chunk = self.mat_emm[ymin:ymax, xmin:xmax]
                self.exthandler.addToTable(
                    [f"a{self.boxNum}", np.min(chunk), np.max(chunk), np.mean(chunk)]
                )
                self.exthandler.addRects([[xmin, xmax, ymin, ymax]])
                pygame.draw.lines(self.overlays, (255, 255, 255), True, points, 3)
                self.renderText(
                    self.overlays, f"a{self.boxNum}", (xmin + 12, ymin + 10)
                )

        if mode == "colorMap":
            self.colorMap = args[0]

            minVal = np.min(self.mat)
            delVal = np.max(self.mat) - minVal
            self.cbarVals = [minVal + i * delVal / 4 for i in range(5)][::-1]

            cbar = np.row_stack(20 * (np.arange(256),))[:, ::-1].astype(np.float32)

            self.image = ThermalImageHelpers.cmap_matplotlib(self.mat, args[0])
            cbar = ThermalImageHelpers.cmap_matplotlib(cbar, args[0])

            self.imsurf = pygame.Surface((WINDOW_SIZE[0] - 245, 512))
            self.imsurf.blit(
                pygame.surfarray.make_surface(
                    np.transpose(self.image[..., ::-1], (1, 0, 2))
                ),
                (0, 0),
            )
            self.imsurf.blit(pygame.surfarray.make_surface(cbar[..., ::-1]), (663, 85))
            for i, val in enumerate(self.cbarVals):
                self.imsurf.blit(
                    self.fonts[0].render(f"{val:.1f}", 1, (255, 255, 255)),
                    (690, 75 + i * 65),
                )
            self.imsurf.blit(
                self.fonts[0].render("\N{DEGREE SIGN}" + "C", 1, (255, 255, 255)),
                (660, 60),
            )
            self.imsurf.blit(self.logo, (658, 405))

        if mode == "scale":
            if args[0] == "reset":
                self.mat = self.mat_emm.copy()
                self.work("colorMap", self.colorMap)

            if args[0] == "switchMode":
                self.managers["scale"].buttons[0][0].set_text(
                    f"Switch to {self.areaMode} mode"
                )
                self.areaMode = "poly" if self.areaMode == "rect" else "rect"
                self.changeMode("scale")

            if args[0] == "scale":

                chunk = self.mat_emm[
                    ThermalImageHelpers.coordinates_in_poly(
                        [(x - 245, y - 15) for x, y in self.linePoints], self.raw.shape
                    )
                ]

                if len(chunk) > 0:
                    self.mat = np.clip(self.mat_emm, np.min(chunk), np.max(chunk))
                    self.work("colorMap", self.colorMap)

        if mode == "emissivity":
            if args[0] == "update":
                emmissivity = self.managers["emissivity"].fields[0][0].get_text()
                ref_temp = self.managers["emissivity"].fields[1][0].get_text()
                atm_temp = self.managers["emissivity"].fields[2][0].get_text()

                np_indices = ThermalImageHelpers.coordinates_in_poly(
                    [(x - 245, y - 15) for x, y in self.linePoints], self.raw.shape
                )
                self.mat_emm[
                    np_indices
                ] = ThermalImageHelpers.change_emissivity_for_roi(
                    thermal_np=None,
                    meta=self.meta,
                    roi_contours=None,
                    raw_roi_values=self.raw[np_indices],
                    indices=None,
                    new_emissivity=emmissivity,
                    ref_temperature=ref_temp,
                    atm_temperature=atm_temp,
                    np_indices=True,
                )

            if args[0] == "reset":
                self.mat_emm = self.mat_orig.copy()

            self.work("scale", "reset")

    def process(self, event):
        """Process input event."""
        self.mx, self.my = self.cursor_rect.center = pygame.mouse.get_pos()
        self.cx = Window.clip(self.mx, 245, 884)
        self.cy = Window.clip(self.my, 15, 526)
        self.cursor_in = (245 < self.mx < 885) and (15 < self.my < 527)
        self.managers[self.mode].process_events(event)
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.cursor_in:
                if self.mode == "line":
                    if len(self.linePoints) < 2:
                        self.linePoints.append((self.mx, self.my))
                if (
                    self.mode == "scale" and self.areaMode == "poly"
                ) or self.mode == "emissivity":
                    if self.selectionComplete:
                        self.linePoints = []
                        self.selectionComplete = False
                    self.linePoints.append((self.mx, self.my))
                    if pygame.key.get_mods() & pygame.KMOD_CTRL:
                        if len(self.linePoints) > 2:
                            self.selectionComplete = True
                if (
                    self.mode == "scale" and self.areaMode == "rect"
                ) or self.mode == "area":
                    self.changeMode(self.mode)
                    self.linePoints.append((self.mx, self.my))

                if self.mode == "spot":
                    self.work("spot")

        if event.type == pygame.MOUSEBUTTONUP:
            if (
                self.mode == "scale" and self.areaMode == "rect"
            ) or self.mode == "area":
                if len(self.linePoints) == 1:
                    self.linePoints.append((self.cx, self.linePoints[0][1]))
                    self.linePoints.append((self.cx, self.cy))
                    self.linePoints.append((self.linePoints[0][0], self.cy))
                    self.selectionComplete = True

    def update(self, time_del):
        """Update events."""
        self.managers[self.mode].update(time_del)

    def draw(self, surface):
        """Draw contents on screen."""
        surface.blit(self.background, (0, 0))
        surface.blit(self.imsurf, (245, 15))
        surface.blit(self.overlays, (245, 15))

        pygame.draw.rect(surface, (255, 255, 255), (245, 540, 760, 35), 1)
        self.managers[self.mode].draw_ui(surface)
        surface.blit(
            self.fonts[1].render(
                f"x:{self.cx - 245:03}   y:{self.cy - 15:03}   temp:{self.mat_emm[self.cy - 15, self.cx - 245]:.4f}",
                1,
                (255, 255, 255),
            ),
            (253, 544),
        )

        if self.mode == "line":
            if len(self.linePoints) == 1:
                pygame.draw.line(
                    surface, (255, 255, 255), self.linePoints[0], (self.cx, self.cy), 3
                )
            if len(self.linePoints) == 2:
                pygame.draw.line(
                    surface, (255, 255, 255), self.linePoints[0], self.linePoints[1], 3
                )

        if (
            self.mode == "scale" and self.areaMode == "poly"
        ) or self.mode == "emissivity":
            if len(self.linePoints) > 0:
                pygame.draw.lines(
                    surface,
                    (255, 255, 255),
                    self.selectionComplete,
                    self.linePoints
                    + ([] if self.selectionComplete else [(self.cx, self.cy)]),
                    3,
                )
        if (self.mode == "scale" and self.areaMode == "rect") or self.mode == "area":
            if not self.selectionComplete:
                if len(self.linePoints) > 0:
                    pygame.draw.lines(
                        surface,
                        (255, 255, 255),
                        True,
                        self.linePoints
                        + [
                            (self.cx, self.linePoints[0][1]),
                            (self.cx, self.cy),
                            (self.linePoints[0][0], self.cy),
                        ],
                        3,
                    )
            else:
                pygame.draw.lines(surface, (255, 255, 255), True, self.linePoints, 3)

        surface.blit(self.cursors[self.cursor_in], self.cursor_rect)


if __name__ == "__main__":
    pygame.mouse.set_visible(False)

    pygame.display.set_caption("Detect Thermal Image Analysis Tool")
    pygame.display.set_icon(pygame.image.load("./assets/icon_gray.png"))
    surface = pygame.display.set_mode(WINDOW_SIZE)
    surface.blit(Window.fonts[2].render("Loading...", 1, (255, 255, 255)), (460, 275))
    pygame.display.update()

    clock = pygame.time.Clock()

    done = False
    NEW_FILE = True
    window = None

    while not done:

        if NEW_FILE:
            filename = openImage()

            if filename:
                surface.fill((0, 0, 0))
                surface.blit(
                    Window.fonts[2].render("Loading...", 1, (255, 255, 255)), (460, 275)
                )
                pygame.display.update()
                newwindow = None
                try:
                    if filename.split(".")[-1] == "pkl":
                        newwindow = Window(filename=filename)
                    else:
                        try:
                            image = ThermalImage(filename, camera_manufacturer="FLIR")
                        except Exception:
                            image = ThermalImage(filename, camera_manufacturer="DJI")
                        newwindow = Window(thermal_image=image)
                except Exception as err:
                    print(f"Exception: {err}")
                    showwarning(title="Error", message="Invalid file selected.")
                if newwindow is not None:
                    if window is not None:
                        window.exthandler.killThreads()
                    window = newwindow
            if not window:
                sys.exit(0)
            NEW_FILE = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == ord("s"):
                    index = 0
                    while os.path.isfile(f"{index}.png"):
                        index += 1
                    pygame.image.save(surface, f"{index}.png")
                    print(f"Saved {index}.png")

            window.process(event)

        window.update(clock.tick(60) / 1000.0)
        window.draw(surface)

        pygame.display.update()

    # For the threads to close before end of program
    window.exthandler.killThreads()
