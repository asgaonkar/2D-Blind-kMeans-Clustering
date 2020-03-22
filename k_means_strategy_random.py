# Import Dependencies
from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import os
from datetime import datetime


# Load Data
mat = loadmat('AllSamples.mat')
mdata = mat['AllSamples']

# Convert to data frame
df = pd.DataFrame(mdata)
df = df.rename(columns={0: 'x', 1: 'y'})
max_val = [max(df['x'])+2, max(df['y'])+2]

# Initial Assignment Function


def assignment(df, centroids):
    for i in centroids.keys():
        # sqrt((x1 - x2)^2 - (y1 - y2)^2)
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['x'] - centroids[i][0]) ** 2
                + (df['y'] - centroids[i][1]) ** 2
            )**2
        )
    centroid_distance_cols = [
        'distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['distance'] = df.loc[:, centroid_distance_cols].min(axis=1)
    df['closest'] = df['closest'].map(
        lambda x: int(x.lstrip('distance_from_')))

    df['color'] = df['closest'].map(lambda x: colmap[x])
    return df


colmap = {1: '#222222', 2: '#ff7c00', 3: '#023eff', 4: '#e8000b', 5: '#8b2be2',
          6: '#1ac938', 7: '#f14cc1', 8: '#a3a3a3', 9: '#ffc400', 10: '#00d7ff'}

os.mkdir("Random/")


for graph_round in range(2):

    obj_k = []
    obj_dist = []

    for k in range(2, 11):

        obj_k.append(k)

        # Randomomize Centroids
        np.random.seed(datetime.now().microsecond)
        centroids = {
            i+1: mdata[np.random.randint(mdata.shape[0]), :]
            for i in range(k)
        }

        fig = plt.figure(figsize=(5, 5))
        fname = "Random/" + str(k) + "-Cluster " + \
            "Initial Random Centroid Assignment"
        plt.title(fname)
        plt.scatter(df['x'], df['y'], color='k')
        for i in centroids.keys():
            plt.scatter(*centroids[i], color=colmap[i], marker="o")
        plt.xlim(0, max_val[0])
        plt.ylim(0, max_val[1])
#         plt.show()
        plt.savefig(fname, dpi='figure')
        # Call to initial assignment function
        df = assignment(df, centroids)

        # Update Stage
        old_centroids = copy.deepcopy(centroids)

        # Function Update centroids
        def update(centroids):
            for i in centroids.keys():
                centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
                centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
            return centroids

        # Call Update centroids
        centroids = update(centroids)

        # Repeat Assigment Stage
        df = assignment(df, centroids)

        # Continue until all assigned categories don't change any more
        while True:
            closest_centroids = df['closest'].copy(deep=True)
            centroids = update(centroids)
            df = assignment(df, centroids)
            if closest_centroids.equals(df['closest']):
                break

        fig = plt.figure(figsize=(5, 5))
        fname = "Random/" + str(k) + "-Cluster " + \
            "Final Cluster Assignment (Random Strategy)"
        plt.title(fname)
        plt.scatter(df['x'], df['y'], color=df['color'],
                    alpha=0.75, edgecolor='k', marker="o")
        for i in centroids.keys():
            plt.scatter(*centroids[i], color=colmap[i],
                        alpha=0.75, marker="X", s=100)
            plt.scatter(*centroids[i], color=colmap[i],
                        alpha=0.75, marker="X", s=100)
        plt.xlim(0, max_val[0])
        plt.ylim(0, max_val[1])
#         plt.show()
        plt.savefig(fname, dpi='figure')

        obj_dist.append(df['distance'].sum())

    fig = plt.figure(figsize=(10, 10))
    plt.title("Objective Function (vs) k-Clusters : Random Centroid")
    plt.ylabel('Objective Function')
    plt.xlabel('k-Clusters')
    plt.grid(True)
    for x, y in zip(obj_k, obj_dist):

        label = "({},{:.2f})".format(x, y)

        plt.annotate(label,  # this is the text
                     (x, y),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(10, -15),  # distance from text to points (x,y)
                     ha='right')  # horizontal alignment can be left, right or center
    plt.xlim(0, obj_k[-1]+2)
    plt.ylim(0, obj_dist[0]*1.15)
    plt.plot(obj_k, obj_dist, linestyle='--', marker='o', color='b')
    plt.savefig("Objective Function (vs) k-Clusters - Random Centroid " +
                str(graph_round), dpi='figure')
