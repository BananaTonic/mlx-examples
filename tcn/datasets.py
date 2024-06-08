# Copyright © 2023 Apple Inc.

import os
import requests
import zipfile
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable

def download_and_extract(data_dir="data", url="https://github.com/laiguokun/multivariate-time-series-data/archive/refs/heads/master.zip"):
    """
    Download and extract the dataset.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    zip_path = os.path.join(data_dir, "multivariate_time_series_data.zip")
    if not os.path.exists(zip_path):
        response = requests.get(url)
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

def load_dataset(data_dir="data", dataset_name="exchange_rate"):
    """
    Load the specified dataset.

    :param data_dir: The directory where data is extracted
    :param dataset_name: The name of the dataset to load
    :return: Data as a numpy array
    """
    file_path = os.path.join(data_dir, f"multivariate-time-series-data-master/{dataset_name}/{dataset_name}.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Please check the dataset path.")
    
    data = pd.read_csv(file_path, delimiter="\t", header=None).values
    return data

def data_generator(T, mem_length, b_size):
    """
    Generate data for the copying memory task

    :param T: The total blank time length
    :param mem_length: The length of the memory to be recalled
    :param b_size: The batch size
    :return: Input and target data tensor
    """
    seq = torch.from_numpy(np.random.randint(1, 9, size=(b_size, mem_length))).float()
    zeros = torch.zeros((b_size, T))
    marker = 9 * torch.ones((b_size, mem_length + 1))
    placeholders = torch.zeros((b_size, mem_length))

    x = torch.cat((seq, zeros[:, :-1], marker), 1)
    y = torch.cat((placeholders, zeros, seq), 1).long()

    x, y = Variable(x), Variable(y)
    return x, y

if __name__ == "__main__":
    download_and_extract()
    for dataset in ["exchange_rate", "electricity", "solar", "traffic"]:
        data = load_dataset(dataset_name=dataset)
        print(f"Loaded {dataset} dataset with shape: {data.shape}")

    T = 10
    mem_length = 5
    b_size = 2
    x, y = data_generator(T, mem_length, b_size)
    print(f"Generated data X shape: {x.shape}, Y shape: {y.shape}")
