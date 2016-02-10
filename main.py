__author__ = 'Wiktor'

import os
import pandas as pd
from sklearn.cluster import KMeans
import plotly.offline as py

# idea was to represent football players from TOP4 teams in English Premier League in 3D cluster
# as part of cluster analysis in data mining using Pandas and scikit-learn


def main():
    # getting data frame with players attributes
    players_df = get_players_df()

    # normalizing parameters in data frame to absolute value
    param_columns = list(players_df.columns.values)
    param_columns.remove("Name")
    param_columns.remove("Club")
    players_df[param_columns] = players_df[param_columns].apply(lambda column: normalize_params(column), axis=0)

    #  dividing n observations into k clusters
    k = 5
    players_df = make_clustering(players_df, k)

    # drawing a 3D cluster using plotly package
    draw_3d_cluster(players_df, k, 'footballers_3d-cluster.html')


def normalize_params(column):
    # value represents the distance between the raw score and the mean in units of the standard deviation
    return (column - column.mean()) / column.std()


def get_players_df():
    unique_columns = ["Name", "Club"]

    # getting main attribute names for non-goalkeeper player
    with open(os.path.join(os.getcwd(), "players", "MUFC_adnan_januzaj.txt"), 'r') as f:
        for line in f:
            if line.strip() != '' and "skills" not in line:
                line = line.replace('\n', '').replace('\t', '').replace(' ', '')
                if not line.isdigit():
                    unique_columns.append(line)

    # getting main attribute names for goalkeeper
    with open(os.path.join(os.getcwd(), "players", "MUFC_david_de gea.txt"), 'r') as f:
        for line in f:
            if line.strip() != '' and "skills" not in line:
                line = line.replace('\n', '').replace('\t', '').replace(' ', '')
                if not (line in unique_columns or line.isdigit()):
                    unique_columns.append(line)

    # initiating Pandas data frame
    path_to_files = os.path.join(os.getcwd(), "players")
    number_of_players = len([f for f in os.listdir(path_to_files)])
    players_df = pd.DataFrame(index=list(range(number_of_players)), columns=unique_columns)

    # filling in data frame
    idx = 0
    for subdir, dirs, files in os.walk(path_to_files):
        for file in files:
            filename = file.replace(".txt", "").split("_")
            columns = ["Name", "Club"]
            params = [(filename[1] + " " + filename[2]).title(), filename[0]]
            d = {}
            with open(os.path.join(path_to_files, file), 'r') as f:
                for line in f:
                    if line.strip() != '' and "skills" not in line:
                        line = line.replace('\n', '').replace('\t', '').replace(' ', '')
                        try:
                            params.append(int(line))
                        except ValueError:
                            columns.append(line)
            for col, par in zip(columns, params):
                players_df.ix[idx][col] = par
            idx += 1

    return players_df.fillna(0)


def make_clustering(data_frame, number_of_clusters):
    # initializing KMeans object, computing clustering and transforming X to cluster-distance space
    k_means_model = KMeans(n_clusters=number_of_clusters)
    distances = k_means_model.fit_transform(data_frame.iloc[:, 2:])

    # adding to out data frame information about unit' cluster and distances to every cluster
    data_frame["cluster"] = k_means_model.labels_
    for i in range(number_of_clusters):
        data_frame["dist " + str(i) + " cluster"] = distances[:, i]

    return data_frame


def draw_3d_cluster(data_frame, number_of_clusters, filename):
    data = []

    # colors to represent clusters
    colors = ['rgb(230,25,25)', 'rgb(80,80,250)', 'rgb(80,230,80)', 'rgb(250,200,90)', 'rgb(100,100,100)',
              'rgb(60,250,250)']

    # visualizing players on 3d scatter
    for i in range(len(data_frame)):
        player = dict(
            name=data_frame.iloc[i]["Name"] + '[' + data_frame.iloc[i]["Club"] + ']',
            type="scatter3d",
            mode='markers',
            showlegend=False,
            marker=dict(size=3, color=colors[data_frame.loc[i]['cluster']], line=dict(width=0)),
            opacity=.9,
            x=data_frame[data_frame.index == i]['dist 0 cluster'],
            y=data_frame[data_frame.index == i]['dist 1 cluster'],
            z=data_frame[data_frame.index == i]['dist 2 cluster'],
        )
        data.append(player)

    # visualizing clusters on 3d scatter
    for i in range(number_of_clusters):
        cluster = dict(
            color=colors[i],
            opacity=.2,
            type="mesh3d",
            hoverinfo='none',
            showlegend=True,
            x=data_frame[data_frame['cluster'] == i]['dist 0 cluster'],
            y=data_frame[data_frame['cluster'] == i]['dist 1 cluster'],
            z=data_frame[data_frame['cluster'] == i]['dist 2 cluster'],
        )
        data.append(cluster)

    fig = dict(data=data)
    py.plot(fig, filename=filename)


if __name__ == "__main__":
    main()