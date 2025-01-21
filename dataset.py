import random
import json
import os
from torch.utils.data import Dataset#, DataLoader
import yaml

# Load the configuration file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Access parameters
data_path = config["data"]["synthetic_data_path"]



class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.data, self.labels = self.preprocessing()
        self.transform = transform

    def preprocessing(self):
        # Open and read json
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = [entry["text"] for entry in data]
        labels = [entry["labels"] for entry in data]
        return texts, labels
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    def __len__(self):
        return len(self.data)

    def negative_sampling(batch_labels, all_labels, max_num_negatives=2):
        num_negatives = random.randint(1, max_num_negatives)
        negative_samples = []
        for labels in batch_labels:
            neg = random.sample([l for l in all_labels if l not in labels], num_negatives)
            negative_samples.append(neg)
        return negative_samples


# Example Usage
if __name__ == "__main__":
    custom_dataset = CustomDataset(data_path)
    print(custom_dataset.__getitem__(5))