import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import json

from sklearn.decomposition import PCA



def read_parsed_data(filter=False):

    with open("data/parsed_data.json", "r") as json_file:
        data = json.load(json_file)

    print("Število VSEH simulacij:", len(data))
    print("Filtriram ...")
    print("Odstranim tiste, ki imajo kak nan, kak negativno vrednost ali so vsi enaki")

    new_data = []
    for point in data:
        part = point['part']

        # no nan values, no negative values, no all same value
        if not np.isnan(part).any() and len(set(part)) > 1 and sum([1 for elt in part if elt < 0]) == 0:
            if point not in new_data:
                new_data.append(point)

    print("Število simulacij po filtriranju:", len(new_data))

    return new_data


def generate_X(data, normalize=True):
    """
    """

    parts = [point['part'] for point in data]
    new_parts = []
    if normalize:
        for point in data:
            part = point['part']
            # transform part
            min_index = np.argmin(part)
            if min_index == 426 or min_index == 427:
                min_index = np.argmin(part[:round(0.5*len(part))])

            first_part, second_part = part[min_index:], part[:min_index]
            diff = second_part[0] - first_part[-1]
            part = part[min_index:] + [part[:min_index][j] - diff for j in range(len(part[:min_index]))]


            # part = part[np.argmin(part):] + part[:np.argmin(part)] 
            # normalize
            part = [(part[i] - np.min(part)) / (np.max(part) - np.min(part)) for i in range(len(part))]
            row = point['params'] + part
            # row = part # brez paramsov!
            new_parts.append(row)
    else:
        # ne delaš transformacij?
        new_parts = [point['params'] + point['part'] for point in data]

    df = pd.DataFrame(new_parts)
    df.dropna(inplace=True)

    df.to_csv("machine_learning/pca/pca_data/normalized_pca_data_FINAL.csv")

    return df

if __name__=="__main__":

    data = read_parsed_data()
    df = generate_X(data, normalize=True)

