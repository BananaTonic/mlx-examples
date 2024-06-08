Usage Instructions
Download and Verify Data:
Run datasets.py to download and verify the datasets.

`python datasets.py`

Train the Model:
Run train_tcn.py to train the TCN model using one of the specified datasets (exchange_rate, electricity, solar, traffic).

`python main.py --dataset exchange_rate`

You can specify different datasets and sequence lengths as needed.

`python train_tcn.py --dataset electricity --seq_length 24`

These scripts will allow you to download, preprocess, and train a Temporal Convolutional Network (TCN) on multiple time series datasets. The data_loader.py script handles the data downloading and preprocessing, while the train_tcn.py script defines the model and training process.