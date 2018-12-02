def test_random_word_generator(random_word_generator):
    """
    0.08432126045227051
    0.8023912906646729
    7.96087908744812
    ----
    1.4066696166992188e-05
    7.390975952148438e-05
    0.00061798095703125
    ----
    0.008105278015136719
    0.008072376251220703
    0.00867319107055664
    """
    t = time.time()
    _ = [random_word_generator.random_word() for _ in range(10)]
    print(time.time() - t)
    t = time.time()
    _ = [random_word_generator.random_word() for _ in range(100)]
    print(time.time() - t)
    t = time.time()
    _ = [random_word_generator.random_word() for _ in range(1000)]
    print(time.time() - t)
    t = time.time()
    _ = [random_word_generator.random_word_optimized() for _ in range(10)]
    print(time.time() - t)
    t = time.time()
    _ = [random_word_generator.random_word_optimized() for _ in range(100)]
    print(time.time() - t)
    t = time.time()
    _ = [random_word_generator.random_word_optimized() for _ in range(1000)]
    print(time.time() - t)
    t = time.time()
    _ = [random_word_generator.random_words(10)]
    print(time.time() - t)
    t = time.time()
    _ = [random_word_generator.random_words(100)]
    print(time.time() - t)
    t = time.time()
    _ = [random_word_generator.random_words(1000)]
    print(time.time() - t)

def embedder_test():
    """Testing the performance of embedder.


    s1000: 5.060769319534302
    s250s*4: 10.278579235076904
    s5000: 7.866734266281128
    s1000s*5: 15.469667434692383
    s10000: 12.214599609375
    s2500s*4: 18.09593439102173
    
    """
    embedder = SentenceEmbedder()
    articles, summaries, scores = load_text_data(size='medium')

    print('all articles:', len(articles))
    # 7,742,028
    all_sents = []
    for article in articles:
        sents = sentence_split(article)
        all_sents.extend(sents)

    print('all sentences:', len(all_sents))
        
    t = time.time()
    # 9.186472415924072
    s1000 = all_sents[:1000]
    embedder.embed(s1000)
    print('s1000:', time.time() - t)

    t = time.time()
    s250s = np.array_split(s1000, 4)
    embedder.embed_list(s250s)
    print('s250s*4:', time.time() - t)
    
    t = time.time()
    # 11.613808870315552
    s5000 = all_sents[:5000]
    embedder.embed(s5000)
    print('s5000:', time.time() - t)

    t = time.time()
    s1000s = np.array_split(s5000, 5)
    embedder.embed_list(s1000s)
    print('s1000s*5:', time.time() - t)

    t = time.time()
    # 15.440065145492554
    s10000 = all_sents[:10000]
    embedder.embed(s10000)
    print('s10000:', time.time() - t)

    t = time.time()
    s2500s = np.array_split(s10000, 4)
    embedder.embed_list(s2500s)
    print('s2500s*4:', time.time() - t)

    t = time.time()
    # 49.14992094039917
    s50000 = all_sents[:50000]
    embedder.embed(s50000)
    print('s50000:', time.time() - t)

    t = time.time()
    s10000s = np.array_split(s50000, 5)
    embedder.embed_list(s10000s)
    print('s10000s*5:', time.time() - t)

    return

