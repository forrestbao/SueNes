import os
import subprocess

def tokenize_stories(stories_dir, tokenized_stories_dir):
    """Maps a whole directory of .story files to a tokenized version using
Stanford CoreNLP Tokenizer"""
    if not os.path.exists(tokenized_stories_dir):
        os.makedirs(tokenized_stories_dir)
    stories = os.listdir(stories_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping.txt", "w") as f:
        for s in stories:
            f.write("%s \t %s\n"
                    % (os.path.join(stories_dir, s),
                       os.path.join(tokenized_stories_dir, s)))
    command = ['java', 'edu.stanford.nlp.process.PTBTokenizer',
               '-ioFileList', '-preserveLines', 'mapping.txt']
    print("Tokenizing %i files in %s and saving in %s..."
          % (len(stories), stories_dir, tokenized_stories_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping.txt")

    # Check that the tokenized stories directory contains the same
    # number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    print("Successfully finished tokenizing %s to %s.\n"
          % (stories_dir, tokenized_stories_dir))

if __name__ == '__test__':
    data_dir = '/home/hebi/mnt/data/nlp/'
    cnn_stories_dir = os.path.join(data_dir, 'cnn/stories')
    cnn_tokenized_stories_dir = os.path.join(data_dir,
                                             'cnn_tokenized_stories')
    dailymail_stories_dir = os.path.join(data_dir, 'dailymail/stories')
    dailymail_tokenized_stories_dir = os.path.join(data_dir,
                                                   'dailymail_tokenized_stories')
    tokenize_stories(cnn_stories_dir, cnn_tokenized_stories_dir)
    tokenize_stories(dailymail_stories_dir, dailymail_tokenized_stories_dir)
