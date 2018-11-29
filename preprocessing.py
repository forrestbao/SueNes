#!/usr/bin/env python3

def read_text_file(text_file):
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines

def get_art_abs(story_file):
    lines = read_text_file(story_file)
    lines = [line.lower() for line in lines]
    # lines = [fix_missing_period(line) for line in lines]
    article_lines = []
    highlights = []
    next_is_highlight = False
    for idx,line in enumerate(lines):
        if line == "":
            continue # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    article = ' '.join(article_lines)
    abstract = ' '.join(highlights)
    return article, abstract

def delete_words(summary, ratio):
    words = summary.split(' ')
    length = len(words)
    indices = set(random.sample(range(length),
                                int((1 - ratio) * length)))
    return ' '.join([words[i] for i in range(length)
                     if i not in indices])


def add_words(summary, ratio, vocab):
    words = summary.split(' ')
    length = len(words)
    indices = set([random.randint(0, length)
                   for _ in range(int((1 - ratio) * length))])
    res = []
    for i in range(length):
        if i in indices:
            res.append(vocab.random_word())
        res.append(words[i])
    return ' '.join(res)

# TODO multi thread generation
# TODO use fake sentence instead of mutating sentence, simulating word2vec loss
# TODO use summary only to see if the article is useful at all
def mutate_summary(summary, vocab):
    """I need to generate random mutation to the summary. Save it to a
    file so that I use the same generated data. For each summary, I
    generate several data:
        
    1. generate 10 random float numbers [0,1] as ratios
    2. for each ratio, do:
    2.1 deletion: select ratio percent of words to remove
    2.2 addition: add ratio percent of new words (from vocab.txt) to
    random places

    Issues:
    
    - should I add better, regularized noise, e.g. gaussian noise? How
      to do that?
    - should I check if the sentence is really modified?
    - should we use the text from original article?
    - should we treat sentences? should we maintain the sentence
      separator period?

    """
    ratios = [random.random() for _ in range(10)]
    res = []
    # add the original summary
    res.append([summary, 1.0, 'orig'])
    # the format: ((summary, score, mutation_method))
    for r in ratios:
        s = delete_words(summary, r)
        res.append((s, r, 'del'))
        s = add_words(summary, r, vocab)
        res.append((s, r, 'add'))
    return res

def preprocess_data():
    """
    1. load stories
    2. tokenize
    3. separate article and summary
    4. chunk and save

    This runs pretty slow
    """
    print('Doing nothing.')
    return 0
    vocab = Vocab(vocab_file, 200000)

    # 92,579 stories
    stories = os.listdir(cnn_tokenized_dir)
    hebi_dir = os.path.join(cnndm_dir, 'hebi')
    if not os.path.exists(hebi_dir):
        os.makedirs(hebi_dir)
    # hebi/xxxxxx/article.txt
    # hebi/xxxxxx/summary.json
    ct = 0
    for s in stories:
        ct += 1
        # if ct > 10:
        #     return
        # print('--', ct)
        if ct % 100 == 0:
            print ('--', ct*100)
        f = os.path.join(cnn_tokenized_dir, s)
        article, summary = get_art_abs(f)
        pairs = mutate_summary(summary, vocab)
        # write down to file
        d = os.path.join(hebi_dir, s)
        if not os.path.exists(d):
            os.makedirs(d)
        article_f = os.path.join(d, 'article.txt')
        summary_f = os.path.join(d, 'summary.json')
        with open(article_f, 'w') as fout:
            fout.write(article)
        with open(summary_f, 'w') as fout:
            json.dump(pairs, fout, indent=4)
