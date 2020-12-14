# AntiRouge


# Install packages

```
pip install --user -r requirements.txt
```

You'll also need to make sure java is installed for running corenlp tokenizers.

<!-- All the rest packages and datasets will be downloaded automatically, including
- corenlp jar file
- cnn/dm dataset

All downloaded assets will be put into DATA_DIR which defaults to `~/data`. -->

# APIs

```py
from antirouge import config
from antirouge.data import get_separate_generators
from antirouge.pre import process_tsv_folder
from antirouge.model import train_model, create_FC_model, create_LSTM_model, create_CNN1D_model
```

# Run

First run `pre/sample_generation.py` to download and preprocess data. See `pre` folder for details.

A sample notebook using the APIs is provided in [main.ipynb](./main.ipynb).
We will be showing the key code below for reference as well.

1. pre-embed the data
```py
from antirouge.pre import process_tsv_folder
process_tsv_folder('pre/cnn_dailymail_add', 'USE', batch_size=1000)
```

2. load the data

```py
from antirouge.data import get_separate_generators
# USE embedded input
train_folder = os.path.join('pre/cnn_dailymail_add', 'train.tsv_embed', 'USE')
test_folder = os.path.join('pre/cnn_dailymail_add', 'test.tsv_embed', 'USE')
validation_folder = os.path.join('pre/cnn_dailymail_add', 'validation.tsv_embed', 'USE')
data_iters = get_separate_generators(train_folder, validation_folder, test_folder, bsize=8)
```

3. create and train model

```py
model = create_FC_model(512)
train_model(model, data_iters)
```