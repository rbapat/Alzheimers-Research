from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

class TrainGrapher:
    def __init__(self, should_graph, *graphs):
        self.should_graph = should_graph
        self.line_shapes = ['-b', '-r', '-g']

        if self.should_graph:
            self.fig, self.graphs = plt.subplots(1, len(graphs))

            if len(graphs) == 1:
                self.graphs = [self.graphs]
            self.graph_dict = {}

    def add_lines(self, graph, graph_loc, *lines):
        if not self.should_graph:
            return [[]] * len(lines)

        line_struct = []
        for i, title in enumerate(lines):
            shape = self.line_shapes[i % len(self.line_shapes)]
            line_struct.append((title, [], shape))

        self.graph_dict[graph] = (line_struct, graph_loc)

        return [i[1] for i in line_struct]

    def update(self):
        if not self.should_graph:
            return

        # TODO: do i need to set_text every time?
        for i, graph_key in enumerate(self.graph_dict):
            self.graphs[i].cla()

            for line in self.graph_dict[graph_key][0]:
                self.graphs[i].plot([i + 1 for i in range(len(line[1]))], line[1], line[2], label = line[0])

            self.graphs[i].legend(loc = self.graph_dict[graph_key][1])
            self.graphs[i].title.set_text(graph_key)

        plt.xlabel("Epoch Number")
        plt.pause(0.0001)

    def show(self):
        if not self.should_graph:
            return
            
        plt.show()

class data_utils:
    @staticmethod
    def get_group(df, header, group):
        return df[df[header] == group]

    @staticmethod
    def count_occurences(df, header):
        freq_dict = {}

        for item in df[header]:
            try:
                freq_dict[item] += 1
            except KeyError:
                freq_dict[item] = 1

        return sorted(freq_dict.items(), key = lambda x: x[1], reverse = True)

    @staticmethod
    def get_volumetric_bounds(scan, axis):
        coords = [(-1, -1), (-1, -1), (-1, -1), (-1, -1)]
        for i in range(scan.shape[axis]):
            if axis == 0:
                subscan = scan[i, :, :]
            elif axis == 1:
                subscan = scan[:, i, :]
            elif axis == 2:
                subscan = scan[:, :, i]
            else:
                print("Invalid Axis")
                return

            if np.sum(subscan) > 0:
                region = np.argwhere(subscan)
                
                try:
                    hull = ConvexHull(region)
                except Exception:
                    continue

                x_list, y_list = [region[v, 1] for v in hull.vertices], [region[v, 0] for v in hull.vertices]

                coordx, coordy = [], []
                minx, maxx = np.argmin(x_list), np.argmax(x_list)
                miny, maxy = np.argmin(y_list), np.argmax(y_list)

                if (x_list[minx] < coords[0][0] or coords[0][0] == -1):
                    coords[0] = (x_list[minx], y_list[minx])

                if (x_list[maxx] > coords[1][0] or coords[1][0] == -1):
                    coords[1] = (x_list[maxx], y_list[maxx])

                if (y_list[miny] < coords[2][1] or coords[2][1] == -1):
                    coords[2] = (x_list[miny], y_list[miny])

                if (y_list[maxy] > coords[3][1] or coords[3][1] == -1):
                    coords[3] = (x_list[maxy], y_list[maxy])

        return coords

    @staticmethod
    def render_scan(scan):
        fig, axes = plt.subplots(1, len(scan.shape))
        for i in range(scan.shape[1]):

            for j in range(len(scan.shape)):
                axes[j].cla()

            subscans = [
                        scan[min(i, scan.shape[0]) - 1, :, :],
                        scan[:, min(i, scan.shape[1]) - 1, :],
                        scan[:, :, min(i, scan.shape[2] - 1)]
                        ]

            for j in range(len(scan.shape)):
                axes[j].imshow(subscans[j], cmap="gray", origin="lower")

            plt.pause(0.0001)

        plt.show()

    @staticmethod
    def crop_scan(scan):
        coordX = data_utils.get_volumetric_bounds(scan, 0)
        coordY = data_utils.get_volumetric_bounds(scan, 2)
        coordZ = data_utils.get_volumetric_bounds(scan, 1)

        new_data = scan[coordZ[2][1]:coordZ[3][1], coordY[0][0]:coordY[1][0], coordX[0][0]:coordX[1][0]]    
        return new_data
