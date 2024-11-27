
import torch


data_path = "IRP/data/processed/" + "data2.dataset"

data = torch.load(data_path)
print("Data type below")
print(type(data))  # Type of the data
print(data)        # Inspect contents

