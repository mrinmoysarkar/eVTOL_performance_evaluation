#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon March 28 1:07 AM 2022

@author: mrinmoy sarkar

"""

import pandas as pd
import os
import sys

def main():
    # cluster_path = './Clustering/CLUSTERING/OUTLIER/'
    cluster_path = './Clustering/CLUSTERING/REGULAR/'

    count_less_100 = 0
    count_more_100 = 0

    for root, dirs, files in os.walk(cluster_path):
        for file in files:
            file_path  = os.path.join(root, file)
            df = pd.read_csv(file_path)
            print(df.shape)
            if df.shape[0]<=1000:
                count_less_100 += 1
            else:
                count_more_100 += 1
    print('less 100: {} more 100: {}'.format(count_less_100, count_more_100))


if __name__ == '__main__':
    main()