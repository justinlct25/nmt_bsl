from sentence_transformers import SentenceTransformer, util
def calculate_cos_sim(sentences):
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    sentences_embeddings = model.encode(sentences)
    cos_sim = util.cos_sim(sentences_embeddings, sentences_embeddings)
    return cos_sim

