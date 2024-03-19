import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader


class TrainingDataSet(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        # TODO: we can do further processing where it actually returns you image the way you want it
        image_name = row["img_filename"]
        instruction = row["instruction"]
        point = row["point"]
        return image_name, instruction, point


def get_data_loader(batch_size=32):
    df = pd.read_csv("training_data_repo/seeclick_web_train.csv")
    dataset = TrainingDataSet(df)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader
