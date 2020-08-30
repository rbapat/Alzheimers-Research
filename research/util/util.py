from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# collection of useful data processing functions
class data_utils:
    @staticmethod
    def get_group(df, header, group):
        return df[df[header] == group]

    # counts the occurences of each value of a given header in a pandas DataFrame
    @staticmethod
    def count_occurences(df, header):
        freq_dict = {}

        for item in df[header]:
            try:
                freq_dict[item] += 1
            except KeyError:
                freq_dict[item] = 1

        return sorted(freq_dict.items(), key = lambda x: x[1], reverse = True)

    # gets the 3D bounds of a skull-stripped MRI frame
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

    # renders a 3D mri frame using matplotlib
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

    # crops all the 0 padding out of a skull stripped MRI scan
    @staticmethod
    def crop_scan(scan):
        coordX = data_utils.get_volumetric_bounds(scan, 0)
        coordY = data_utils.get_volumetric_bounds(scan, 2)
        coordZ = data_utils.get_volumetric_bounds(scan, 1)

        new_data = scan[coordZ[2][1]:coordZ[3][1], coordY[0][0]:coordY[1][0], coordX[0][0]:coordX[1][0]]    
        return new_data
