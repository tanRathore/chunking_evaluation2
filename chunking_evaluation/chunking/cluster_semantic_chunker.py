from .base_chunker import BaseChunker
from typing import List
import numpy as np
from chunking_evaluation.chunking import RecursiveTokenChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from chunking_evaluation.utils import get_gemini_embedding_function , gemini_token_count
import chromadb.utils.embedding_functions as embedding_functions

class ClusterSemanticChunker(BaseChunker):
    def __init__(self, embedding_function=None, max_chunk_size=400, min_chunk_size=50, length_function = gemini_token_count):
        self.splitter = RecursiveTokenChunker(
            chunk_size=min_chunk_size,
            chunk_overlap=0,
            length_function=gemini_token_count,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""]
        )

        if embedding_function is None:
            #embedding_function = self._get_gemini_embedding_function()
            embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key="")
        self._chunk_size = max_chunk_size
        self.max_cluster = max_chunk_size // min_chunk_size
        self.embedding_function = embedding_function

    # def _get_gemini_embedding_function(self):
    #     return GoogleGenerativeAIEmbeddings(model="models/embedding-001").embed_documents

    # def _gemini_token_count(self, text: str) -> int:
    #     embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001").embed_documents([text])[0]
    #     return len(embedding)

    def _get_similarity_matrix(self, embedding_function, sentences):
        BATCH_SIZE = 500
        N = len(sentences)
        embedding_matrix = None

        for i in range(0, N, BATCH_SIZE):
            batch_sentences = sentences[i:i + BATCH_SIZE]
            embeddings = embedding_function(batch_sentences)

            # Convert embeddings list of lists to numpy array
            batch_embedding_matrix = np.array(embeddings)

            # Append the batch embedding matrix to the main embedding matrix
            if embedding_matrix is None:
                embedding_matrix = batch_embedding_matrix
            else:
                embedding_matrix = np.concatenate((embedding_matrix, batch_embedding_matrix), axis=0)

        similarity_matrix = np.dot(embedding_matrix, embedding_matrix.T)

        return similarity_matrix

    def _calculate_reward(self, matrix, start, end):
        sub_matrix = matrix[start:end + 1, start:end + 1]
        return np.sum(sub_matrix)

    def _optimal_segmentation(self, matrix, max_cluster_size, window_size=3):
        mean_value = np.mean(matrix[np.triu_indices(matrix.shape[0], k=1)])
        matrix = matrix - mean_value  # Normalize the matrix
        np.fill_diagonal(matrix, 0)  # Set diagonal to 0 to avoid trivial solutions

        n = matrix.shape[0]
        dp = np.zeros(n)
        segmentation = np.zeros(n, dtype=int)

        for i in range(n):
            for size in range(1, max_cluster_size + 1):
                if i - size + 1 >= 0:
                    reward = self._calculate_reward(matrix, i - size + 1, i)
                    adjusted_reward = reward
                    if i - size >= 0:
                        adjusted_reward += dp[i - size]
                    if adjusted_reward > dp[i]:
                        dp[i] = adjusted_reward
                        segmentation[i] = i - size + 1

        clusters = []
        i = n - 1
        while i >= 0:
            start = segmentation[i]
            clusters.append((start, i))
            i = start - 1

        clusters.reverse()
        return clusters

    def split_text(self, text: str) -> List[str]:
        sentences = self.splitter.split_text(text)

        similarity_matrix = self._get_similarity_matrix(self.embedding_function, sentences)

        clusters = self._optimal_segmentation(similarity_matrix, max_cluster_size=self.max_cluster)

        docs = [' '.join(sentences[start:end + 1]) for start, end in clusters]

        return docs