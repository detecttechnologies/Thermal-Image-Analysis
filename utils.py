import csv
import pickle
import threading
from tkinter import Tk, filedialog, messagebox, ttk

import numpy as np
import pygame
from matplotlib import figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import Image


class SaveData:
    pass


class TableView(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.start()
        self.initialized = False
        self.data = []

    def addRow(self, entry):
        while not self.initialized:
            pass
        entry[1:] = [round(ent, 3) for ent in entry[1:]]
        self.treev.insert("", "end", text="L1", values=entry)
        self.data.append(entry)

    def writeToFile(self, filename):
        with open(filename, "w") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow(["Element", "Min", "Max", "Average"])
            for data in self.data:
                writer.writerow(data)

    def killTable(self):
        self.root.quit()
        self.root.update()

    def run(self):
        self.root = Tk()
        self.root.protocol("WM_DELETE_WINDOW", lambda: None)
        self.root.title("Table")

        self.treev = ttk.Treeview(self.root, selectmode="browse")
        self.treev.pack(side="right")
        self.treev.pack(side="right")

        verscrlbar = ttk.Scrollbar(self.root, orient="vertical", command=self.treev.yview)
        verscrlbar.pack(side="right", fill="x")

        self.treev.configure(xscrollcommand=verscrlbar.set)
        self.treev["columns"] = ("1", "2", "3", "4")
        self.treev["show"] = "headings"

        self.treev.column("1", width=100, anchor="c")
        self.treev.column("2", width=100, anchor="se")
        self.treev.column("3", width=100, anchor="se")
        self.treev.column("4", width=100, anchor="se")

        self.treev.heading("1", text="Element")
        self.treev.heading("2", text="min")
        self.treev.heading("3", text="max")
        self.treev.heading("4", text="average")

        self.initialized = True

        self.root.mainloop()


class Figure(threading.Thread):
    def __init__(self, plots):
        threading.Thread.__init__(self)
        self.plots = plots
        self.start()

    def killFigure(self):
        self.root.quit()
        self.root.update()

    def saveFig(self, filename):
        self.fig.savefig(filename)

    def run(self):
        self.root = Tk()
        self.root.protocol("WM_DELETE_WINDOW", lambda: None)
        self.root.title("Plot")

        self.fig = figure.Figure()
        plot = self.fig.add_subplot(111)

        for x, y, label in self.plots:
            plot.plot(x, y, label=label)
        plot.legend()

        canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        canvas.draw()
        canvas.get_tk_widget().pack()

        toolbar = NavigationToolbar2Tk(canvas, self.root)
        toolbar.update()

        canvas.get_tk_widget().pack()

        self.root.mainloop()


class WindowHandler:
    def __init__(self):
        self.mainTable = None
        self.mainFigure = None
        self.plots = []
        self.killed = False

    def __del__(self):
        if not self.killed:
            self.killThreads()

    def killThreads(self):
        if self.mainTable:
            self.mainTable.killTable()
            self.mainTable.join()
        if self.mainFigure:
            self.mainFigure.killFigure()
            self.mainFigure.join()
        self.killed = True

    def addToTable(self, entry):
        if self.mainTable is None:
            self.mainTable = TableView()
        self.mainTable.addRow(entry)

    def loadGraph(self, plots_in):
        self.plots = plots_in
        if self.plots:
            self.mainFigure = Figure(self.plots)

    def linePlot(self, mat, label, startPoint, endPoint, resolution=100, interpolation="bilinear"):

        direcion = endPoint - startPoint
        distance = np.linalg.norm(direcion)
        distances = []
        values = []

        for i in range(resolution):

            point = startPoint + i * direcion / resolution

            if interpolation == "bilinear":
                x = int(point[0])
                y = int(point[1])
                dx = point[0] - x
                dy = point[1] - y
                val = (np.array([[1 - dx, dx]]) @ mat[x : x + 2, y : y + 2] @ np.array([[1 - dy], [dy]]))[0, 0]

            if interpolation == "nearest_neighbour":
                val = mat[round(point[0]), round(point[1])]

            values.append(val)
            distances.append(i * distance / resolution)

        self.addToTable([label, min(values), max(values), sum(values) / len(values)])

        if self.mainFigure:
            self.mainFigure.killFigure()
        self.plots.append([distances, values, label])
        self.mainFigure = Figure(self.plots)


def saveImage(window):
    imageSurface = window.imsurf
    overlays = window.overlays
    exthandler = window.exthandler
    Tk().withdraw()
    file = filedialog.asksaveasfile()

    if file:

        imgdata = pygame.surfarray.pixels3d(imageSurface).astype(np.uint8)
        imgdata = np.swapaxes(imgdata, 0, 1)

        filename = file.name
        try:
            Image.fromarray(imgdata).save(filename)
        except ValueError:
            filename += ".png"
            Image.fromarray(imgdata).save(filename)

        if messagebox.askquestion("Save options", "Do you want to save with the lines?") == "yes":
            imageSurface.blit(overlays, (0, 0))

            data = SaveData()
            data.plots = window.exthandler.plots
            data.tableEntries = []
            data.mat = window.mat
            data.mat_orig = window.mat_orig
            data.mat_emm = window.mat_emm
            data.raw = window.raw
            data.meta = window.meta
            data.overlays = pygame.image.tostring(overlays, "RGBA")

            if exthandler.mainFigure:
                exthandler.mainFigure.saveFig(".".join(filename.split(".")[:-1]) + "_plot.png")
            if exthandler.mainTable:
                exthandler.mainTable.writeToFile(".".join(filename.split(".")[:-1]) + "_values.csv")
                data.tableEntries = exthandler.mainTable.data

            with open(".".join(filename.split(".")[:-1]) + ".pkl", "wb") as f:
                pickle.dump(data, f, -1)


def openImage():
    Tk().withdraw()
    filename = filedialog.askopenfilename()
    return filename
