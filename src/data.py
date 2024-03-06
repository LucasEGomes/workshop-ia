import os
import tarfile
import zipfile
from pathlib import Path

import pandas as pd
import requests


def stack_overflow_survey(year: int,
                          data_dir: str = "./",
                          destiny: str = "./") -> int:
    if not os.path.exists(data_dir):
        return 2
    url = f"https://info.stackoverflowsolutions.com/rs/719-EMH-566/images/stack-overflow-developer-survey-{year}.zip"
    if year == 2023:
        url = "https://cdn.stackoverflow.co/files/jo7n4k8s/production/49915bfd46d0902c3564fd9a06b509d08a20488c.zip/stack-overflow-developer-survey-2023.zip"
    response = requests.get(url, timeout=60, verify=False)
    if not response.ok:
        return 1
    filename = f"stack-overflow-developer-survey-{year}.zip"
    filename = os.path.join(data_dir, filename)
    with open(filename, "wb") as f:
        f.write(response.content)
    return _extract(filename, destiny)


def _extract(filename: str, destiny: str) -> int:
    with zipfile.ZipFile(filename) as z:
        z.extractall(path=destiny)
    return 0



def load_housing_data(data_dir: str = "./"):
    tarball_path = os.path.join(data_dir, "housing.tgz")
    if not os.path.exists(tarball_path):
        os.makedirs(data_dir, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        response = requests.get(url, timeout=60, verify=False)
        with open(tarball_path, "wb") as f:
            f.write(response.content)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path=data_dir)
    return pd.read_csv(os.path.join(data_dir, "housing", "housing.csv"))


def load_mnist(data_dir: str = "./"):
    filename = os.path.join(data_dir, "mnist.csv")
    if not os.path.exists(filename):
        url = "https://raw.githubusercontent.com/sbussmann/kaggle-mnist/master/Data/train.csv"
        response = requests.get(url, timeout=60, verify=False)
        with open(filename, "wb") as f:
            f.write(response.content)
    return pd.read_csv(filename)
