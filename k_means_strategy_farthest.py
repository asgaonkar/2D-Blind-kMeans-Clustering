# Import Dependencies
from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
import os
from datetime import datetime

# Load Data
mat = loadmat('AllSamples.mat')
mdata = mat['AllSamples']

# Convert to data frame
df = pd.DataFrame(mdata)
df = df.rename(columns={0: 'x', 1: 'y'})
max_val = [max(df['x'])+2, max(df['y'])+2]

os.mkdir("Farthest/")

# function to compute euclidean distance


def distance(p1, p2):
    return np.sum((p1 - p2)**2)

# Centroid Initialization


def seeds(data, k):

    centroids = []
    centroids.append(data[np.random.randint(
        data.shape[0]), :])

    for _ in range(k - 1):

        dist = []
        for i in range(data.shape[0]):
            point = data[i, :]
            d = sys.maxsize

            for j in range(len(centroids)):
                temp_dist = distance(point, centroids[j])
                d = min(d, temp_dist)
            dist.append(d)

        # select data point with maximum distance as our next centroid
        dist = np.array(dist)
        next_centroid = data[np.argmax(dist), :]
        centroids.append(next_centroid)
        dist = []

        dicts = {}
    k = 0
    for i in centroids:
        dicts[k+1] = [i[0], i[1]]
        k += 1

    centroids = dicts

    return centroids

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


for graph_round in range(2):

    obj_k = []
    obj_dist = []

    for k in range(2, 11):

        obj_k.append(k)

        # Randomomize Centroids
        np.random.seed(datetime.now().microsecond)

        centroids = seeds(mdata, k)

        fig = plt.figure(figsize=(5, 5))
        fname = "Farthest/" + str(k) + "-Cluster " + \
            "Initial Random Centroid Assignment"
        plt.title(fname)
        plt.scatter(df['x'], df['y'], color='k')
        for i in centroids.keys():
            plt.scatter(*centroids[i], color=colmap[i])
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
        fname = "Farthest/" + str(k) + "-Cluster " + "Final Cluster Assignment"
        plt.title(fname)
        plt.scatter(df['x'], df['y'], color=df['color'],
                    alpha=0.75, edgecolor='k')
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
    plt.title("Objective Function (vs) k-Clusters")
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
    plt.plot(obj_k, obj_dist, linestyle='--', marker='o', color='g')
    plt.savefig("Objective Function (vs) k-Clusters - Farthest Centroid " +
                str(graph_round), dpi='figure')
