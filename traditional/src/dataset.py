import os
from PIL import Image
from torch.utils.data import Dataset

class OCRDataset(Dataset):
    def __init__(self, image_dir, label_path, transform=None):
        self.image_dir = image_dir
        self.label_path = label_path
        self.transform = transform
        self.samples = []

        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                image_name, text = line.split("\t", 1)
                self.samples.append((image_name, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_name, text = self.samples[index]
        image_path = os.path.join(self.image_dir, image_name)

        image = Image.open(image_path).convert("L")

        if self.transform is not None:
            image = self.transform(image)

        return image, text