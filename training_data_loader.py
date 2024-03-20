import pandas as pd
import os
import numpy as np
from PIL import Image
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
        img = Image.open(
            os.path.join(
                "training_data/cpfs01/user/chengkanzhi/seeclick_web_imgs_part/",
                image_name,
            )
        )
        img = img.convert("RGB")
        img = np.array(img)
        # TODO: Probably should do resizing so batch in data loader is of same shape
        # Tested, and it still was but as a precaution.
        instruction = row["instruction"]
        # point = row["point"]
        label = row["label"]
        return img, instruction, label


def get_data_loader(batch_size=32):
    df = pd.read_csv("training_data_repo/seeclick_web_train.csv")
    dataset = TrainingDataSet(df)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


# data = get_data_loader()

# for i, (img, instruction, point) in enumerate(data):
#     print(img.shape)
#     break
