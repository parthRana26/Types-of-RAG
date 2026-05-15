# from sentence_transformers import SentenceTransformer

# # 1. Load a pretrained Sentence Transformer model
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# # The sentences to encode
# sentences = [
#     "The weather is lovely today.",
#     "It's so sunny outside!",
#     "He drove to the stadium.",
# ]

# # 2. Calculate embeddings by calling model.encode()
# embeddings = model.encode(sentences)
# print(embeddings.shape)
# # [3, 384]

# # 3. Calculate the embedding similarities
# similarities = model.similarity(embeddings, embeddings)
# print(similarities)

# Install via pip
# pip install -U sentence-transformers

from sentence_transformers import SentenceTransformer
sentences = ["This"]
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
print(embeddings.shape)
print(embeddings)