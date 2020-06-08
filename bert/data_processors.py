import csv
from utils import DataProcessor
from utils import InputExample
import tokenization
import os


class Cnn30kProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""
  def __init__(self):
    self.f_train = 'neg-train2.tsv'
    self.f_test = 'neg-test2.tsv'
    self.f_predict = 'TAC2010_test.tsv'

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, self.f_train)), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, self.f_test)), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, self.f_predict)), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      # if i == 0:
      #   continue
      guid = "%s-%s" % (set_type, i)
      line0 = ' '.join(line[0].split()[0:400])
      line1 = ' '.join(line[1].split()[0:200])
      text_a = tokenization.convert_to_unicode(line0)
      text_b = tokenization.convert_to_unicode(line1)
      label = tokenization.convert_to_unicode(line[2])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class CnnMutationProcessor(DataProcessor):
  def __init__(self):
    name = 'add'
    self.f_train = 'mutate-%s-train2.tsv' % name
    self.f_train = 'mutate-%s-test2.tsv' % name
    self.f_test = 'mutate-%s-test2.tsv' % name

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, self.f_train)), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, self.f_test)), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, self.f_test)), "test")

  def get_labels(self):
    """See base class."""
    #return ["0", "1"]
    return None

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      # if i == 0:
      #   continue
      guid = "%s-%s" % (set_type, i)
      try:
        line0 = ' '.join(line[0].split()[0:400])
        line1 = ' '.join(line[1].split()[0:200])
        text_a = tokenization.convert_to_unicode(line0)
        text_b = tokenization.convert_to_unicode(line1)
        #label = float(line[2].strip()) 
        label = tokenization.convert_to_unicode(line[2])
      except:
        print(line)
        print('0', line[0])
        print('1', line[1])
      
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class CnnMutationDelProcessor(CnnMutationProcessor):
    def __init__(self):
        self.f_train = 'mutate-delete-test2.tsv'
        self.f_test = 'mutate-delete-test2.tsv'


class CnnMutationRepProcessor(CnnMutationProcessor):
    def __init__(self):
        self.f_train = 'mutate-replace-test2.tsv'
        self.f_test = 'mutate-replace-test2.tsv'


