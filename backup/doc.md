
## Download and process the data

Download data (TODO automate this). Currently you can download from
[[https://cs.nyu.edu/~kcho/DMQA/][CNN/DM benchmark]]. Unzip and put =cnn= and
=dm= directories into the DATA_DIR folder which defaults to `~/data`.

The preprocess includes:
- tokenization
- preprocess story and summary files
- embed stories using sentence embedding

```py
from antirouge import config
from antirouge import pre

pre.tokenize_stories(config.CNN_DIR, config.CNN_TOKENIZED_DIR)

pre.serial_process_story(config.CNN_TOKENIZED_DIR,
                            os.path.join(config.CNN_SERIAL_DIR, 'story'))

pre.serial_process_embed(config.CNN_SERIAL_DIR, 'USE')
pre.serial_process_embed(config.CNN_SERIAL_DIR, 'USE-Large')
pre.serial_process_embed(config.CNN_SERIAL_DIR, 'InferSent')
```

## Train and evaluate the model

```py
def exp_word():
    print('Glove exp')
    folder = os.path.join(config.CNN_SERIAL_DIR, 'story')
    model = get_word_model()
    train_model(model, folder)

def exp_USE():
    print('USE exp')
    folder = os.path.join(config.CNN_SERIAL_DIR, 'USE')
    model = get_sent_model(512)
    train_model(model, folder)
    
def exp_USE_Large():
    print('USE Large exp')
    folder = os.path.join(config.CNN_SERIAL_DIR, 'USE-Large')
    model = get_sent_model(512)
    train_model(model, folder)

def exp_InferSent():
    print('InferSent exp')
    folder = os.path.join(config.CNN_SERIAL_DIR, 'InferSent')
    model = get_sent_model(4096)
    train_model(model, folder)
```