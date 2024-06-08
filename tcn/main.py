# Copyright © 2023 Apple Inc.

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import data_generator, download_and_extract, load_dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length]
        y = self.data[idx+self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1, self.conv2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def train_model(args):
    # Set parameters
    seq_length = args.seq_length
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.001

    # Download and load dataset
    download_and_extract()
    data = load_dataset(dataset_name=args.dataset)

    # Prepare dataset and dataloader
    dataset = TimeSeriesDataset(data, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define model, loss function, and optimizer
    input_channels = 1
    output_size = data.shape[1]
    num_channels = [25, 25, 25, 25]
    kernel_size = 7
    dropout = 0.05

    model = TCN(input_channels, num_channels, kernel_size=kernel_size, dropout=dropout)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for i, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.unsqueeze(1)  # Change shape to (batch_size, 1, seq_length)
            outputs = model(inputs)
            loss = criterion(outputs[:, :, -1], targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}')

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a TCN on Time Series Data.")
    parser.add_argument("--dataset", type=str, required=True, choices=["exchange_rate", "electricity", "solar", "traffic"], help="The dataset to use.")
    parser.add_argument("--seq_length", type=int, default=24, help="Sequence length for the time series data.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available.")
    args = parser.parse_args()
    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    train_model(args)
