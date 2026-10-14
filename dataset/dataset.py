from PIL import Image
from torch.utils.data import Dataset
import pandas as pd


class CustomDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(row["jpg"]).convert("RGB")
        mask = Image.open(row["mask"]).convert("L")
        text = row["txt"]
        class_id = row["class_id"]

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
            img = img * 2 - 1
            mask = (mask > 0.5).float()

        return {
            "jpg": img,
            "txt": text,
            "class_id": class_id,
            "mask": mask,
        }
