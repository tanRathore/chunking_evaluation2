# from typing import Callable
# from chunking_evaluation.utils import rigorous_document_search, get_gemini_embedding_function
# import os
# import pandas as pd
# import json
# import chromadb
# import numpy as np
# from typing import List
# from importlib import resources
# import chromadb.utils.embedding_functions as embedding_functions
# from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction

# def sum_of_ranges(ranges):
#     return sum(end - start for start, end in ranges)


# def union_ranges(ranges):
#     # Sort ranges based on the starting index
#     sorted_ranges = sorted(ranges, key=lambda x: x[0])
    
#     # Initialize with the first range
#     merged_ranges = [sorted_ranges[0]]
    
#     for current_start, current_end in sorted_ranges[1:]:
#         last_start, last_end = merged_ranges[-1]
        
#         # Check if the current range overlaps or is contiguous with the last range in the merged list
#         if current_start <= last_end:
#             # Merge the two ranges
#             merged_ranges[-1] = (last_start, max(last_end, current_end))
#         else:
#             # No overlap, add the current range as new
#             merged_ranges.append((current_start, current_end))
    
#     return merged_ranges


# def intersect_two_ranges(range1, range2):
#     # Unpack the ranges
#     start1, end1 = range1
#     start2, end2 = range2
    
#     # Calculate the maximum of the starting indices and the minimum of the ending indices
#     intersect_start = max(start1, start2)
#     intersect_end = min(end1, end2)
    
#     # Check if the intersection is valid (the start is less than or equal to the end)
#     if intersect_start <= intersect_end:
#         return (intersect_start, intersect_end)
#     else:
#         return None  # Return None if there is no intersection
    

# # Define the difference function
# def difference(ranges, target):
#     """
#     Takes a set of ranges and a target range, and returns the difference.
    
#     Args:
#     - ranges (list of tuples): A list of tuples representing ranges. Each tuple is (a, b) where a <= b.
#     - target (tuple): A tuple representing a target range (c, d) where c <= d.
    
#     Returns:
#     - List of tuples representing ranges after removing the segments that overlap with the target range.
#     """
#     result = []
#     target_start, target_end = target

#     for start, end in ranges:
#         if end < target_start or start > target_end:
#             # No overlap
#             result.append((start, end))
#         elif start < target_start and end > target_end:
#             # Target is a subset of this range, split it into two ranges
#             result.append((start, target_start))
#             result.append((target_end, end))
#         elif start < target_start:
#             # Overlap at the start
#             result.append((start, target_start))
#         elif end > target_end:
#             # Overlap at the end
#             result.append((target_end, end))
#         # Else, this range is fully contained by the target, and is thus removed

#     return result


# def find_target_in_document(document, target):
#     start_index = document.find(target)
#     if start_index == -1:
#         return None
#     end_index = start_index + len(target)
#     return start_index, end_index


# class BaseEvaluation:
#     def __init__(self, questions_csv_path: str, chroma_db_path=None, corpora_id_paths=None):
#         self.corpora_id_paths = corpora_id_paths
#         self.questions_csv_path = questions_csv_path
#         self.corpus_list = []
#         self._load_questions_df()

#         if chroma_db_path is not None:
#             self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
#         else:
#             self.chroma_client = chromadb.Client()

#         self.is_general = False

#     def _load_questions_df(self):
#         if os.path.exists(self.questions_csv_path):
#             self.questions_df = pd.read_csv(self.questions_csv_path)
#             self.questions_df['references'] = self.questions_df['references'].apply(json.loads)
#         else:
#             self.questions_df = pd.DataFrame(columns=['question', 'references', 'corpus_id'])
        
#         self.corpus_list = self.questions_df['corpus_id'].unique().tolist()

#     # def _get_chunks_and_metadata(self, splitter):
#     #     documents = []
#     #     metadatas = []
#     #     for corpus_id in self.corpus_list:
#     #         corpus_path = corpus_id
#     #         if self.corpora_id_paths is not None:
#     #             corpus_path = self.corpora_id_paths[corpus_id]

#     #         with open(corpus_path, 'r') as file:
#     #             corpus = file.read()

#     #         current_documents = splitter.split_text(corpus)
#     #         current_metadatas = []
#     #         for document in current_documents:
#     #             try:
#     #                 _, start_index, end_index = rigorous_document_search(corpus, document)
#     #             except:
#     #                 print(f"Error in finding {document} in {corpus_id}")
#     #                 raise Exception(f"Error in finding {document} in {corpus_id}")
#     #             current_metadatas.append({"start_index": start_index, "end_index": end_index, "corpus_id": corpus_id})
#     #         documents.extend(current_documents)
#     #         metadatas.extend(current_metadatas)
#     #     return documents, metadatas
#     def _get_chunks_and_metadata(self, splitter):
#         import logging  # Ensure logging is imported at the top of your file
#         logging.basicConfig(level=logging.DEBUG)
        
#         documents = []
#         metadatas = []
#         for corpus_id in self.corpus_list:
#             corpus_path = corpus_id
#             if self.corpora_id_paths is not None:
#                 corpus_path = self.corpora_id_paths[corpus_id]

#             logging.debug(f"Processing corpus_id: {corpus_id}, corpus_path: {corpus_path}")
            
#             with open(corpus_path, 'r') as file:
#                 corpus = file.read()

#             current_documents = splitter.split_text(corpus)
#             current_metadatas = []
#             for document in current_documents:
#                 try:
#                     _, start_index, end_index = rigorous_document_search(corpus, document)
#                 except:
#                     print(f"Error in finding {document} in {corpus_id}")
#                     raise Exception(f"Error in finding {document} in {corpus_id}")
#                 current_metadatas.append({"start_index": start_index, "end_index": end_index, "corpus_id": corpus_id})
#             documents.extend(current_documents)
#             metadatas.extend(current_metadatas)
#         return documents, metadatas

#     def _full_precision_score(self, chunk_metadatas):
#         ioc_scores = []
#         recall_scores = []
#         highlighted_chunks_count = []

#         for index, row in self.questions_df.iterrows():
#             question = row['question']
#             references = row['references']
#             corpus_id = row['corpus_id']

#             ioc_score = 0
#             numerator_sets = []
#             denominator_chunks_sets = []
#             unused_highlights = [(x['start_index'], x['end_index']) for x in references]

#             highlighted_chunk_count = 0

#             for metadata in chunk_metadatas:
#                 chunk_start, chunk_end, chunk_corpus_id = metadata['start_index'], metadata['end_index'], metadata['corpus_id']

#                 if chunk_corpus_id != corpus_id:
#                     continue
                
#                 contains_highlight = False

#                 for ref_obj in references:
#                     reference = ref_obj['content']
#                     ref_start, ref_end = int(ref_obj['start_index']), int(ref_obj['end_index'])
#                     intersection = intersect_two_ranges((chunk_start, chunk_end), (ref_start, ref_end))
                    
#                     if intersection is not None:
#                         contains_highlight = True
#                         unused_highlights = difference(unused_highlights, intersection)
#                         numerator_sets = union_ranges([intersection] + numerator_sets)
#                         denominator_chunks_sets = union_ranges([(chunk_start, chunk_end)] + denominator_chunks_sets)
                
#                 if contains_highlight:
#                     highlighted_chunk_count += 1
                
#             highlighted_chunks_count.append(highlighted_chunk_count)
#             denominator_sets = union_ranges(denominator_chunks_sets + unused_highlights)
            
#             if numerator_sets:
#                 ioc_score = sum_of_ranges(numerator_sets) / sum_of_ranges(denominator_sets)
            
#             ioc_scores.append(ioc_score)
#             recall_score = 1 - (sum_of_ranges(unused_highlights) / sum_of_ranges([(x['start_index'], x['end_index']) for x in references]))
#             recall_scores.append(recall_score)

#         return ioc_scores, highlighted_chunks_count

#     def _scores_from_dataset_and_retrievals(self, question_metadatas, highlighted_chunks_count):
#         iou_scores = []
#         recall_scores = []
#         precision_scores = []
#         for (index, row), highlighted_chunk_count, metadatas in zip(self.questions_df.iterrows(), highlighted_chunks_count, question_metadatas):
#             question = row['question']
#             references = row['references']
#             corpus_id = row['corpus_id']

#             numerator_sets = []
#             denominator_chunks_sets = []
#             unused_highlights = [(x['start_index'], x['end_index']) for x in references]

#             for metadata in metadatas[:highlighted_chunk_count]:
#                 chunk_start, chunk_end, chunk_corpus_id = metadata['start_index'], metadata['end_index'], metadata['corpus_id']

#                 if chunk_corpus_id != corpus_id:
#                     continue
                
#                 for ref_obj in references:
#                     reference = ref_obj['content']
#                     ref_start, ref_end = int(ref_obj['start_index']), int(ref_obj['end_index'])
#                     intersection = intersect_two_ranges((chunk_start, chunk_end), (ref_start, ref_end))
                    
#                     if intersection is not None:
#                         unused_highlights = difference(unused_highlights, intersection)
#                         numerator_sets = union_ranges([intersection] + numerator_sets)
#                         denominator_chunks_sets = union_ranges([(chunk_start, chunk_end)] + denominator_chunks_sets)
            
#             if numerator_sets:
#                 numerator_value = sum_of_ranges(numerator_sets)
#             else:
#                 numerator_value = 0

#             recall_denominator = sum_of_ranges([(x['start_index'], x['end_index']) for x in references])
#             precision_denominator = sum_of_ranges([(x['start_index'], x['end_index']) for x in metadatas[:highlighted_chunk_count]])
#             iou_denominator = precision_denominator + sum_of_ranges(unused_highlights)

#             recall_score = numerator_value / recall_denominator
#             recall_scores.append(recall_score)

#             precision_score = numerator_value / precision_denominator
#             precision_scores.append(precision_score)

#             iou_score = numerator_value / iou_denominator
#             iou_scores.append(iou_score)

#         return iou_scores, recall_scores, precision_scores

#     def _chunker_to_collection(self, chunker, embedding_function, chroma_db_path: str = None, collection_name: str = None):
#         collection = None

#         if chroma_db_path is not None:
#             try:
#                 chunk_client = chromadb.PersistentClient(path=chroma_db_path)
#                 collection = chunk_client.create_collection(collection_name, embedding_function=embedding_function, metadata={"hnsw:search_ef": 50})
#                 print("Created collection: ", collection_name)
#             except Exception as e:
#                 print("Failed to create collection: ", e)
#                 pass

#         collection_name = "auto_chunk"
#         if collection is None:
#             try:
#                 self.chroma_client.delete_collection(collection_name)
#             except ValueError as e:
#                 pass
#             collection = self.chroma_client.create_collection(collection_name, embedding_function=embedding_function, metadata={"hnsw:search_ef": 50})

#         docs, metas = self._get_chunks_and_metadata(chunker)

#         BATCH_SIZE = 500
#         for i in range(0, len(docs), BATCH_SIZE):
#             batch_docs = docs[i:i + BATCH_SIZE]
#             batch_metas = metas[i:i + BATCH_SIZE]
#             batch_ids = [str(i) for i in range(i, i + len(batch_docs))]
#             collection.add(
#                 documents=batch_docs,
#                 metadatas=batch_metas,
#                 ids=batch_ids
#             )

#         return collection

#     def _convert_question_references_to_json(self):
#         def safe_json_loads(row):
#             try:
#                 return json.loads(row)
#             except:
#                 pass

#         self.questions_df['references'] = self.questions_df['references'].apply(safe_json_loads)

#     def run(self, chunker, embedding_function=None, retrieve: int = 5, db_to_save_chunks: str = None):
#         """
#         This function runs the evaluation over the provided chunker.

#         Parameters:
#         chunker: The chunker to evaluate.
#         embedding_function: The embedding function to use for calculating the nearest neighbours during the retrieval step. If not provided, the default Gemini embedding function is used.
#         retrieve: The number of chunks to retrieve per question. If set to -1, the function will retrieve the minimum number of chunks that contain excerpts for a given query. This is typically around 1 to 3 but can vary by question. By setting a specific value for retrieve, this number is fixed for all queries.
#         """
#         self._load_questions_df()
#         if embedding_function is None:
#             #embedding_function = get_gemini_embedding_function()
#             embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key="")


#         collection = None
#         if db_to_save_chunks is not None:
#             chunk_size = chunker._chunk_size if hasattr(chunker, '_chunk_size') else "0"
#             chunk_overlap = chunker._chunk_overlap if hasattr(chunker, '_chunk_overlap') else "0"
#             embedding_function_name = embedding_function.__class__.__name__
#             if embedding_function_name == "SentenceTransformerEmbeddingFunction":
#                 embedding_function_name = "SentEmbFunc"
#             collection_name = embedding_function_name + '_' + chunker.__class__.__name__ + '_' + str(int(chunk_size)) + '_' + str(int(chunk_overlap))
#             try:
#                 chunk_client = chromadb.PersistentClient(path=db_to_save_chunks)
#                 collection = chunk_client.get_collection(collection_name, embedding_function=embedding_function)
#             except Exception as e:
#                 collection = self._chunker_to_collection(chunker, embedding_function, chroma_db_path=db_to_save_chunks, collection_name=collection_name)

#         if collection is None:
#             collection = self._chunker_to_collection(chunker, embedding_function)

#         question_collection = None

#         if self.is_general:
#             with resources.as_file(resources.files('chunking_evaluation.evaluation_framework') / 'general_evaluation_data') as general_benchmark_path:
#                 questions_client = chromadb.PersistentClient(path=os.path.join(general_benchmark_path, 'questions_db'))
#                 #if embedding_function.__class__.__name__ == "GeminiEmbeddingFunction":
#                 try:
#                         #if embedding_function._model_name == "models/embedding-001":
#                     question_collection = questions_client.get_collection("auto_questions_gemini_large", embedding_function=embedding_function)
#                 except Exception as e:
#                         print("Warning: Failed to use the frozen embeddings originally used in the paper. As a result, this package will now generate a new set of embeddings. The change should be minimal and only come from the noise floor of Gemini's embedding function. The error: ", e)
        
#         if not self.is_general or question_collection is None:
#             try:
#                 self.chroma_client.delete_collection("auto_questions")
#             except ValueError as e:
#                 pass
#             question_collection = self.chroma_client.create_collection("auto_questions", embedding_function=embedding_function, metadata={"hnsw:search_ef": 50})
#             question_collection.add(
#                 documents=self.questions_df['question'].tolist(),
#                 metadatas=[{"corpus_id": x} for x in self.questions_df['corpus_id'].tolist()],
#                 ids=[str(i) for i in self.questions_df.index]
#             )
        
#         question_db = question_collection.get(include=['embeddings'])

#         # Convert ids to integers for sorting
#         question_db['ids'] = [int(id) for id in question_db['ids']]
#         # Sort both ids and embeddings based on ids
#         _, sorted_embeddings = zip(*sorted(zip(question_db['ids'], question_db['embeddings'])))

#         # Sort questions_df in ascending order
#         self.questions_df = self.questions_df.sort_index()

#         brute_iou_scores, highlighted_chunks_count = self._full_precision_score(collection.get()['metadatas'])

#         if retrieve == -1:
#             maximum_n = min(20, max(highlighted_chunks_count))
#         else:
#             highlighted_chunks_count = [retrieve] * len(highlighted_chunks_count)
#             maximum_n = retrieve

#         retrievals = collection.query(query_embeddings=list(sorted_embeddings), n_results=maximum_n)

#         iou_scores, recall_scores, precision_scores = self._scores_from_dataset_and_retrievals(retrievals['metadatas'], highlighted_chunks_count)

#         corpora_scores = {}
#         for index, row in self.questions_df.iterrows():
#             if row['corpus_id'] not in corpora_scores:
#                 corpora_scores[row['corpus_id']] = {
#                     "precision_omega_scores": [],
#                     "iou_scores": [],
#                     "recall_scores": [],
#                     "precision_scores": []
#                 }
            
#             corpora_scores[row['corpus_id']]['precision_omega_scores'].append(brute_iou_scores[index])
#             corpora_scores[row['corpus_id']]['iou_scores'].append(iou_scores[index])
#             corpora_scores[row['corpus_id']]['recall_scores'].append(recall_scores[index])
#             corpora_scores[row['corpus_id']]['precision_scores'].append(precision_scores[index])


#         brute_iou_mean = np.mean(brute_iou_scores)
#         brute_iou_std = np.std(brute_iou_scores)

#         recall_mean = np.mean(recall_scores)
#         recall_std = np.std(recall_scores)

#         iou_mean = np.mean(iou_scores)
#         iou_std = np.std(iou_scores)

#         precision_mean = np.mean(precision_scores)
#         precision_std = np.std(precision_scores)

#         # print("Recall scores: ", recall_scores)
#         # print("Precision scores: ", precision_scores)
#         # print("Recall Mean: ", recall_mean)
#         # print("Precision Mean: ", precision_mean)

#         return {
#             "corpora_scores": corpora_scores,
#             "iou_mean": iou_mean,
#             "iou_std": iou_std,
#             "recall_mean": recall_mean,
#             "recall_std": recall_std,
#             "precision_omega_mean": brute_iou_mean,
#             "precision_omega_std": brute_iou_std,
#             "precision_mean": precision_mean,
#             "precision_std": precision_std
#         

from typing import Callable
from chunking_evaluation.utils import rigorous_document_search, get_gemini_embedding_function
import os
import pandas as pd
import json
import chromadb
import numpy as np
from typing import List
from importlib import resources
import chromadb.utils.embedding_functions as embedding_functions
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction

def sum_of_ranges(ranges):
    return sum(end - start for start, end in ranges)


def union_ranges(ranges):
    # Sort ranges based on the starting index
    sorted_ranges = sorted(ranges, key=lambda x: x[0])
    
    # Initialize with the first range
    merged_ranges = [sorted_ranges[0]]
    
    for current_start, current_end in sorted_ranges[1:]:
        last_start, last_end = merged_ranges[-1]
        
        # Check if the current range overlaps or is contiguous with the last range in the merged list
        if current_start <= last_end:
            # Merge the two ranges
            merged_ranges[-1] = (last_start, max(last_end, current_end))
        else:
            # No overlap, add the current range as new
            merged_ranges.append((current_start, current_end))
    
    return merged_ranges


def intersect_two_ranges(range1, range2):
    # Unpack the ranges
    start1, end1 = range1
    start2, end2 = range2
    
    # Calculate the maximum of the starting indices and the minimum of the ending indices
    intersect_start = max(start1, start2)
    intersect_end = min(end1, end2)
    
    # Check if the intersection is valid (the start is less than or equal to the end)
    if intersect_start <= intersect_end:
        return (intersect_start, intersect_end)
    else:
        return None  # Return None if there is no intersection
    

def difference(ranges, target):
    """
    Takes a set of ranges and a target range, and returns the difference.
    """
    result = []
    target_start, target_end = target

    for start, end in ranges:
        if end < target_start or start > target_end:
            # No overlap
            result.append((start, end))
        elif start < target_start and end > target_end:
            # Target is a subset of this range, split it into two ranges
            result.append((start, target_start))
            result.append((target_end, end))
        elif start < target_start:
            # Overlap at the start
            result.append((start, target_start))
        elif end > target_end:
            # Overlap at the end
            result.append((target_end, end))
        # Else, this range is fully contained by the target, and is thus removed

    return result


def find_target_in_document(document, target):
    start_index = document.find(target)
    if start_index == -1:
        return None
    end_index = start_index + len(target)
    return start_index, end_index


def add_to_chromadb_in_batches(collection, documents, metadatas, ids, max_payload_size=10000):
    """
    Add documents to ChromaDB in smaller batches, ensuring each batch does not exceed the specified payload size.
    """
    batch_documents = []
    batch_metadatas = []
    batch_ids = []
    current_batch_size = 0

    for doc, meta, id_ in zip(documents, metadatas, ids):
        # Calculate the size of this item
        doc_size = len(doc.encode('utf-8'))
        meta_size = 0
        for k, v in meta.items():
            meta_size += len(str(k).encode('utf-8')) + len(str(v).encode('utf-8'))
        id_size = len(id_.encode('utf-8'))

        item_size = doc_size + meta_size + id_size

        # Check if adding this item would exceed the max payload size
        if current_batch_size + item_size > max_payload_size:
            # Add the current batch to ChromaDB
            if batch_documents:
                collection.add(
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
            # Reset batch
            batch_documents = [doc]
            batch_metadatas = [meta]
            batch_ids = [id_]
            current_batch_size = item_size
        else:
            # Add this item to the current batch
            batch_documents.append(doc)
            batch_metadatas.append(meta)
            batch_ids.append(id_)
            current_batch_size += item_size

    # Add any remaining items in the last batch
    if batch_documents:
        collection.add(
            documents=batch_documents,
            metadatas=batch_metadatas,
            ids=batch_ids
        )


class BaseEvaluation:
    def __init__(self, questions_csv_path: str, chroma_db_path=None, corpora_id_paths=None):
        self.corpora_id_paths = corpora_id_paths
        self.questions_csv_path = questions_csv_path
        self.corpus_list = []
        self._load_questions_df()

        if chroma_db_path is not None:
            self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        else:
            self.chroma_client = chromadb.Client()

        self.is_general = False

    def _load_questions_df(self):
        if os.path.exists(self.questions_csv_path):
            self.questions_df = pd.read_csv(self.questions_csv_path)
            self.questions_df['references'] = self.questions_df['references'].apply(json.loads)
        else:
            self.questions_df = pd.DataFrame(columns=['question', 'references', 'corpus_id'])
        
        self.corpus_list = self.questions_df['corpus_id'].unique().tolist()

    def _get_chunks_and_metadata(self, splitter):
        import logging
        logging.basicConfig(level=logging.DEBUG)
        
        documents = []
        metadatas = []
        for corpus_id in self.corpus_list:
            corpus_path = corpus_id
            if self.corpora_id_paths is not None:
                corpus_path = self.corpora_id_paths[corpus_id]

            logging.debug(f"Processing corpus_id: {corpus_id}, corpus_path: {corpus_path}")
            
            with open(corpus_path, 'r') as file:
                corpus = file.read()

            current_documents = splitter.split_text(corpus)
            current_metadatas = []
            for document in current_documents:
                try:
                    _, start_index, end_index = rigorous_document_search(corpus, document)
                except:
                    print(f"Error in finding {document} in {corpus_id}")
                    raise Exception(f"Error in finding {document} in {corpus_id}")
                current_metadatas.append({"start_index": start_index, "end_index": end_index, "corpus_id": corpus_id})
            documents.extend(current_documents)
            metadatas.extend(current_metadatas)
        return documents, metadatas

    def _full_precision_score(self, chunk_metadatas):
        ioc_scores = []
        recall_scores = []
        highlighted_chunks_count = []

        for index, row in self.questions_df.iterrows():
            question = row['question']
            references = row['references']
            corpus_id = row['corpus_id']

            ioc_score = 0
            numerator_sets = []
            denominator_chunks_sets = []
            unused_highlights = [(x['start_index'], x['end_index']) for x in references]

            highlighted_chunk_count = 0

            for metadata in chunk_metadatas:
                chunk_start, chunk_end, chunk_corpus_id = metadata['start_index'], metadata['end_index'], metadata['corpus_id']

                if chunk_corpus_id != corpus_id:
                    continue
                
                contains_highlight = False

                for ref_obj in references:
                    reference = ref_obj['content']
                    ref_start, ref_end = int(ref_obj['start_index']), int(ref_obj['end_index'])
                    intersection = intersect_two_ranges((chunk_start, chunk_end), (ref_start, ref_end))
                    
                    if intersection is not None:
                        contains_highlight = True
                        unused_highlights = difference(unused_highlights, intersection)
                        numerator_sets = union_ranges([intersection] + numerator_sets)
                        denominator_chunks_sets = union_ranges([(chunk_start, chunk_end)] + denominator_chunks_sets)
                
                if contains_highlight:
                    highlighted_chunk_count += 1
                
            highlighted_chunks_count.append(highlighted_chunk_count)
            denominator_sets = union_ranges(denominator_chunks_sets + unused_highlights)
            
            if numerator_sets:
                ioc_score = sum_of_ranges(numerator_sets) / sum_of_ranges(denominator_sets)
            
            ioc_scores.append(ioc_score)
            recall_score = 1 - (sum_of_ranges(unused_highlights) / sum_of_ranges([(x['start_index'], x['end_index']) for x in references]))
            recall_scores.append(recall_score)

        return ioc_scores, highlighted_chunks_count

    def _scores_from_dataset_and_retrievals(self, question_metadatas, highlighted_chunks_count):
        iou_scores = []
        recall_scores = []
        precision_scores = []
        for (index, row), highlighted_chunk_count, metadatas in zip(self.questions_df.iterrows(), highlighted_chunks_count, question_metadatas):
            question = row['question']
            references = row['references']
            corpus_id = row['corpus_id']

            numerator_sets = []
            denominator_chunks_sets = []
            unused_highlights = [(x['start_index'], x['end_index']) for x in references]

            for metadata in metadatas[:highlighted_chunk_count]:
                chunk_start, chunk_end, chunk_corpus_id = metadata['start_index'], metadata['end_index'], metadata['corpus_id']

                if chunk_corpus_id != corpus_id:
                    continue
                
                for ref_obj in references:
                    reference = ref_obj['content']
                    ref_start, ref_end = int(ref_obj['start_index']), int(ref_obj['end_index'])
                    intersection = intersect_two_ranges((chunk_start, chunk_end), (ref_start, ref_end))
                    
                    if intersection is not None:
                        unused_highlights = difference(unused_highlights, intersection)
                        numerator_sets = union_ranges([intersection] + numerator_sets)
                        denominator_chunks_sets = union_ranges([(chunk_start, chunk_end)] + denominator_chunks_sets)
            
            if numerator_sets:
                numerator_value = sum_of_ranges(numerator_sets)
            else:
                numerator_value = 0

            recall_denominator = sum_of_ranges([(x['start_index'], x['end_index']) for x in references])
            precision_denominator = sum_of_ranges([(x['start_index'], x['end_index']) for x in metadatas[:highlighted_chunk_count]])
            iou_denominator = precision_denominator + sum_of_ranges(unused_highlights)

            recall_score = numerator_value / recall_denominator
            recall_scores.append(recall_score)

            precision_score = numerator_value / precision_denominator
            precision_scores.append(precision_score)

            iou_score = numerator_value / iou_denominator
            iou_scores.append(iou_score)

        return iou_scores, recall_scores, precision_scores

    def _chunker_to_collection(self, chunker, embedding_function, chroma_db_path: str = None, collection_name: str = None):
        collection = None

        if chroma_db_path is not None:
            try:
                chunk_client = chromadb.PersistentClient(path=chroma_db_path)
                collection = chunk_client.create_collection(collection_name, embedding_function=embedding_function, metadata={"hnsw:search_ef": 50})
                print("Created collection: ", collection_name)
            except Exception as e:
                print("Failed to create collection: ", e)
                pass

        collection_name = "auto_chunk"
        if collection is None:
            try:
                self.chroma_client.delete_collection(collection_name)
            except ValueError as e:
                pass
            collection = self.chroma_client.create_collection(collection_name, embedding_function=embedding_function, metadata={"hnsw:search_ef": 50})

        docs, metas = self._get_chunks_and_metadata(chunker)

        # Instead of a fixed batch size and a loop, we use our batching function
        all_ids = [str(idx) for idx in range(len(docs))]
        add_to_chromadb_in_batches(collection, docs, metas, all_ids, max_payload_size=10000)

        return collection

    def _convert_question_references_to_json(self):
        def safe_json_loads(row):
            try:
                return json.loads(row)
            except:
                pass

        self.questions_df['references'] = self.questions_df['references'].apply(safe_json_loads)

    def run(self, chunker, embedding_function=None, retrieve: int = 5, db_to_save_chunks: str = None):
        self._load_questions_df()
        if embedding_function is None:
            embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key="")

        collection = None
        if db_to_save_chunks is not None:
            chunk_size = chunker._chunk_size if hasattr(chunker, '_chunk_size') else "0"
            chunk_overlap = chunker._chunk_overlap if hasattr(chunker, '_chunk_overlap') else "0"
            embedding_function_name = embedding_function.__class__.__name__
            if embedding_function_name == "SentenceTransformerEmbeddingFunction":
                embedding_function_name = "SentEmbFunc"
            collection_name = embedding_function_name + '_' + chunker.__class__.__name__ + '_' + str(int(chunk_size)) + '_' + str(int(chunk_overlap))
            try:
                chunk_client = chromadb.PersistentClient(path=db_to_save_chunks)
                collection = chunk_client.get_collection(collection_name, embedding_function=embedding_function)
            except Exception as e:
                collection = self._chunker_to_collection(chunker, embedding_function, chroma_db_path=db_to_save_chunks, collection_name=collection_name)

        if collection is None:
            collection = self._chunker_to_collection(chunker, embedding_function)

        question_collection = None

        if self.is_general:
            with resources.as_file(resources.files('chunking_evaluation.evaluation_framework') / 'general_evaluation_data') as general_benchmark_path:
                questions_client = chromadb.PersistentClient(path=os.path.join(general_benchmark_path, 'questions_db'))
                try:
                    question_collection = questions_client.get_collection("auto_questions_gemini_large", embedding_function=embedding_function)
                except Exception as e:
                    print("Warning: Failed to use the frozen embeddings. Now generating a new set of embeddings with Gemini. Error:", e)
        
        if not self.is_general or question_collection is None:
            try:
                self.chroma_client.delete_collection("auto_questions")
            except ValueError as e:
                pass
            question_collection = self.chroma_client.create_collection("auto_questions", embedding_function=embedding_function, metadata={"hnsw:search_ef": 50})
            question_collection.add(
                documents=self.questions_df['question'].tolist(),
                metadatas=[{"corpus_id": x} for x in self.questions_df['corpus_id'].tolist()],
                ids=[str(i) for i in self.questions_df.index]
            )
        
        question_db = question_collection.get(include=['embeddings'])

        # Convert ids to integers for sorting
        question_db['ids'] = [int(id) for id in question_db['ids']]
        # Sort both ids and embeddings based on ids
        _, sorted_embeddings = zip(*sorted(zip(question_db['ids'], question_db['embeddings'])))

        # Sort questions_df in ascending order
        self.questions_df = self.questions_df.sort_index()

        brute_iou_scores, highlighted_chunks_count = self._full_precision_score(collection.get()['metadatas'])

        if retrieve == -1:
            maximum_n = min(20, max(highlighted_chunks_count))
        else:
            highlighted_chunks_count = [retrieve] * len(highlighted_chunks_count)
            maximum_n = retrieve

        retrievals = collection.query(query_embeddings=list(sorted_embeddings), n_results=maximum_n)

        iou_scores, recall_scores, precision_scores = self._scores_from_dataset_and_retrievals(retrievals['metadatas'], highlighted_chunks_count)

        corpora_scores = {}
        for index, row in self.questions_df.iterrows():
            if row['corpus_id'] not in corpora_scores:
                corpora_scores[row['corpus_id']] = {
                    "precision_omega_scores": [],
                    "iou_scores": [],
                    "recall_scores": [],
                    "precision_scores": []
                }
            
            corpora_scores[row['corpus_id']]['precision_omega_scores'].append(brute_iou_scores[index])
            corpora_scores[row['corpus_id']]['iou_scores'].append(iou_scores[index])
            corpora_scores[row['corpus_id']]['recall_scores'].append(recall_scores[index])
            corpora_scores[row['corpus_id']]['precision_scores'].append(precision_scores[index])


        brute_iou_mean = np.mean(brute_iou_scores)
        brute_iou_std = np.std(brute_iou_scores)

        recall_mean = np.mean(recall_scores)
        recall_std = np.std(recall_scores)

        iou_mean = np.mean(iou_scores)
        iou_std = np.std(iou_scores)

        precision_mean = np.mean(precision_scores)
        precision_std = np.std(precision_scores)

        return {
            "corpora_scores": corpora_scores,
            "iou_mean": iou_mean,
            "iou_std": iou_std,
            "recall_mean": recall_mean,
            "recall_std": recall_std,
            "precision_omega_mean": brute_iou_mean,
            "precision_omega_std": brute_iou_std,
            "precision_mean": precision_mean,
            "precision_std": precision_std
        }