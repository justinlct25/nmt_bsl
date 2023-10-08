from gensim.models import KeyedVectors

# Load the binary model
model = KeyedVectors.load_word2vec_format('en.wiki.bpe.vs100000.d300.w2v.bin', binary=True)


# Save it in text format
model.save_word2vec_format('en.wiki.bpe.vs100000.d300.w2v.txt', binary=False)
