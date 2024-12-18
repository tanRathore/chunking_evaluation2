# from abc import ABC, abstractmethod
# from enum import Enum
# import logging
# from typing import (
#     AbstractSet,
#     Any,
#     Callable,
#     Collection,
#     Iterable,
#     List,
#     Literal,
#     Optional,
#     Sequence,
#     Type,
#     TypeVar,
#     Union,
# )
# from .base_chunker import BaseChunker
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from attr import dataclass

# logger = logging.getLogger(__name__)

# TS = TypeVar("TS", bound="TextSplitter")

# class TextSplitter(BaseChunker, ABC):
#     """Interface for splitting text into chunks."""

#     def __init__(
#         self,
#         chunk_size: int = 4000,
#         chunk_overlap: int = 200,
#         length_function: Callable[[str], int] = len,
#         keep_separator: bool = False,
#         add_start_index: bool = False,
#         strip_whitespace: bool = True,
#     ) -> None:
#         """Create a new TextSplitter.

#         Args:
#             chunk_size: Maximum size of chunks to return
#             chunk_overlap: Overlap in characters between chunks
#             length_function: Function that measures the length of given chunks
#             keep_separator: Whether to keep the separator in the chunks
#             add_start_index: If `True`, includes chunk's start index in metadata
#             strip_whitespace: If `True`, strips whitespace from the start and end of
#                               every document
#         """
#         if chunk_overlap > chunk_size:
#             raise ValueError(
#                 f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
#                 f"({chunk_size}), should be smaller."
#             )
#         self._chunk_size = chunk_size
#         self._chunk_overlap = chunk_overlap
#         self._length_function = length_function
#         self._keep_separator = keep_separator
#         self._add_start_index = add_start_index
#         self._strip_whitespace = strip_whitespace

#     @abstractmethod
#     def split_text(self, text: str) -> List[str]:
#         """Split text into multiple components."""

#     def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
#         text = separator.join(docs)
#         if self._strip_whitespace:
#             text = text.strip()
#         if text == "":
#             return None
#         else:
#             return text

#     def _merge_splits(self, splits: Iterable[str], separator: str) -> List[str]:
#         # We now want to combine these smaller pieces into medium size
#         # chunks to send to the LLM.
#         separator_len = self._length_function(separator)

#         docs = []
#         current_doc: List[str] = []
#         total = 0
#         for d in splits:
#             _len = self._length_function(d)
#             if (
#                 total + _len + (separator_len if len(current_doc) > 0 else 0)
#                 > self._chunk_size
#             ):
#                 if total > self._chunk_size:
#                     logger.warning(
#                         f"Created a chunk of size {total}, "
#                         f"which is longer than the specified {self._chunk_size}"
#                     )
#                 if len(current_doc) > 0:
#                     doc = self._join_docs(current_doc, separator)
#                     if doc is not None:
#                         docs.append(doc)
#                     # Keep on popping if:
#                     # - we have a larger chunk than in the chunk overlap
#                     # - or if we still have any chunks and the length is long
#                     while total > self._chunk_overlap or (
#                         total + _len + (separator_len if len(current_doc) > 0 else 0)
#                         > self._chunk_size
#                         and total > 0
#                     ):
#                         total -= self._length_function(current_doc[0]) + (
#                             separator_len if len(current_doc) > 1 else 0
#                         )
#                         current_doc = current_doc[1:]
#             current_doc.append(d)
#             total += _len + (separator_len if len(current_doc) > 1 else 0)
#         doc = self._join_docs(current_doc, separator)
#         if doc is not None:
#             docs.append(doc)
#         return docs

#     @classmethod
#     def from_gemini_encoder(
#         cls: Type[TS],
#         model_name: str = "models/embedding-001",
#         **kwargs: Any,
#     ) -> TS:
#         """Text splitter that uses Gemini encoder to count length."""
#         embedding_model = GoogleGenerativeAIEmbeddings(model=model_name)

#         def _gemini_encoder(text: str) -> int:
#             embedding = embedding_model.embed_documents([text])[0]
#             return len(embedding)

#         if issubclass(cls, FixedTokenChunker):
#             extra_kwargs = {
#                 "model_name": model_name,
#             }
#             kwargs = {**kwargs, **extra_kwargs}

#         return cls(length_function=_gemini_encoder, **kwargs)

# class FixedTokenChunker(TextSplitter):
#     """Splitting text to tokens using Gemini embeddings."""

#     def __init__(
#         self,
#         model_name: str = "models/embedding-001",
#         chunk_size: int = 4000,
#         chunk_overlap: int = 200,
#         **kwargs: Any,
#     ) -> None:
#         """Create a new FixedTokenChunker."""
#         super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
#         self.embedding_model = GoogleGenerativeAIEmbeddings(model=model_name)

#     def split_text(self, text: str) -> List[str]:
#         def _encode(_text: str) -> List[int]:
#             embedding = self.embedding_model.embed_documents([_text])[0]
#             return list(range(len(embedding)))  # Use the length of embedding as token ids

#         tokenizer = Tokenizer(
#             chunk_overlap=self._chunk_overlap,
#             tokens_per_chunk=self._chunk_size,
#             decode=lambda x: text,  # Placeholder for decoding
#             encode=_encode,
#         )

#         return split_text_on_tokens(text=text, tokenizer=tokenizer)

# @dataclass(frozen=True)
# class Tokenizer:
#     """Tokenizer data class."""

#     chunk_overlap: int
#     """Overlap in tokens between chunks"""
#     tokens_per_chunk: int
#     """Maximum number of tokens per chunk"""
#     decode: Callable[[List[int]], str]
#     """ Function to decode a list of token ids to a string"""
#     encode: Callable[[str], List[int]]
#     """ Function to encode a string to a list of token ids"""


# def split_text_on_tokens(*, text: str, tokenizer: Tokenizer) -> List[str]:
#     """Split incoming text and return chunks using tokenizer."""
#     splits: List[str] = []
#     input_ids = tokenizer.encode(text)
#     start_idx = 0
#     cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
#     chunk_ids = input_ids[start_idx:cur_idx]
#     while start_idx < len(input_ids):
#         splits.append(tokenizer.decode(chunk_ids))
#         if cur_idx == len(input_ids):
#             break
#         start_idx += tokenizer.tokens_per_chunk - tokenizer.chunk_overlap
#         cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
#         chunk_ids = input_ids[start_idx:cur_idx]
#     return splits






# from abc import ABC, abstractmethod
# import logging
# from typing import (
#     AbstractSet,
#     Any,
#     Callable,
#     Collection,
#     Iterable,
#     List,
#     Literal,
#     Optional,
#     Type,
#     TypeVar,
#     Union,
# )
# from .base_chunker import BaseChunker
# from langchain_google_genai import GoogleGenerativeAIEmbeddings  
# from langchain_google_genai import ChatGoogleGenerativeAI
# from google.cloud import aiplatform 
# from attr import dataclass
# import sentencepiece
# import os
# from attr import dataclass

# logger = logging.getLogger(__name__)

# TS = TypeVar("TS", bound="TextSplitter")
# class GemmaTokenizer:
#     """Tokenizer class to use Gemma Tokenizer with SentencePiece."""

#     def __init__(self, model_path: Optional[str]):
#         # Load tokenizer model from file
#         assert os.path.isfile(model_path), f"Model file not found: {model_path}"
#         self.sp_model = sentencepiece.SentencePieceProcessor()
#         self.sp_model.Load(model_path)

#         # BOS / EOS token IDs
#         self.n_words: int = self.sp_model.GetPieceSize()
#         self.bos_id: int = self.sp_model.bos_id()
#         self.eos_id: int = self.sp_model.eos_id()
#         self.pad_id: int = self.sp_model.pad_id()

#     def encode(self, s: str, bos: bool = True, eos: bool = False) -> List[int]:
#         """Converts a string into a list of tokens."""
#         assert isinstance(s, str)
#         tokens = self.sp_model.EncodeAsIds(s)
#         if bos:
#             tokens = [self.bos_id] + tokens
#         if eos:
#             tokens = tokens + [self.eos_id]
#         return tokens

#     def decode(self, tokens: List[int]) -> str:
#         """Converts a list of tokens into a string."""
#         return self.sp_model.DecodeIds(tokens)
# class TextSplitter(BaseChunker, ABC):
#     """Interface for splitting text into chunks."""

#     def __init__(
#         self,
#         chunk_size: int = 4000,
#         chunk_overlap: int = 200,
#         length_function: Callable[[str], int] = len,
#         keep_separator: bool = False,
#         add_start_index: bool = False,
#         strip_whitespace: bool = True,
#     ) -> None:
#         """Create a new TextSplitter."""
#         if chunk_overlap > chunk_size:
#             raise ValueError(
#                 f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
#                 f"({chunk_size}), should be smaller."
#             )
#         self._chunk_size = chunk_size
#         self._chunk_overlap = chunk_overlap
#         self._length_function = length_function
#         self._keep_separator = keep_separator
#         self._add_start_index = add_start_index
#         self._strip_whitespace = strip_whitespace

#     @abstractmethod
#     def split_text(self, text: str) -> List[str]:
#         """Split text into multiple components."""

#     def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
#         text = separator.join(docs)
#         if self._strip_whitespace:
#             text = text.strip()
#         if text == "":
#             return None
#         else:
#             return text

#     def _merge_splits(self, splits: Iterable[str], separator: str) -> List[str]:
#         # Merges smaller pieces into medium-sized chunks.
#         separator_len = self._length_function(separator)
#         docs = []
#         current_doc: List[str] = []
#         total = 0
#         for d in splits:
#             _len = self._length_function(d)
#             if (
#                 total + _len + (separator_len if len(current_doc) > 0 else 0)
#                 > self._chunk_size
#             ):
#                 if total > self._chunk_size:
#                     logger.warning(
#                         f"Created a chunk of size {total}, "
#                         f"which is longer than the specified {self._chunk_size}"
#                     )
#                 if len(current_doc) > 0:
#                     doc = self._join_docs(current_doc, separator)
#                     if doc is not None:
#                         docs.append(doc)
#                     while total > self._chunk_overlap or (
#                         total + _len + (separator_len if len(current_doc) > 0 else 0)
#                         > self._chunk_size
#                         and total > 0
#                     ):
#                         total -= self._length_function(current_doc[0]) + (
#                             separator_len if len(current_doc) > 1 else 0
#                         )
#                         current_doc = current_doc[1:]
#             current_doc.append(d)
#             total += _len + (separator_len if len(current_doc) > 1 else 0)
#         doc = self._join_docs(current_doc, separator)
#         if doc is not None:
#             docs.append(doc)
#         return docs

#     @classmethod
#     def from_tiktoken_encoder(
#         cls: Type[TS],
#         encoding_name: str = "gpt2",
#         model_name: Optional[str] = None,
#         allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
#         disallowed_special: Union[Literal["all"], Collection[str]] = "all",
#         **kwargs: Any,
#     ) -> TS:
#         """Create a TextSplitter that uses tiktoken encoder for token counting."""
#         try:
#             import tiktoken
#         except ImportError:
#             raise ImportError(
#                 "Could not import tiktoken python package. "
#                 "Please install it with `pip install tiktoken`."
#             )

#         if model_name is not None:
#             enc = tiktoken.encoding_for_model(model_name)
#         else:
#             enc = tiktoken.get_encoding(encoding_name)

#         def _tiktoken_encoder(text: str) -> int:
#             return len(
#                 enc.encode(
#                     text,
#                     allowed_special=allowed_special,
#                     disallowed_special=disallowed_special,
#                 )
#             )
#         if issubclass(cls, FixedTokenChunker):
#             extra_kwargs = {
#                 "encoding_name": encoding_name,
#                 "model_name": model_name,
#                 "allowed_special": allowed_special,
#                 "disallowed_special": disallowed_special,
#             }
#             kwargs = {**kwargs, **extra_kwargs}

#         return cls(length_function=_tiktoken_encoder, **kwargs)

#     def from_gemini_tokenizer(
#         cls: Type[TS],
#         model_name: str = "gemini-1.5-flash-002",
#         **kwargs: Any,
#     ) -> TS:
#         """Create a TextSplitter that uses Gemini tokenizer for token counting."""
#         client = aiplatform.gapic.ModelServiceClient()

#         def _gemini_tokenizer(text: str) -> int:
#             response = client.predict(
#                 name=model_name,
#                 instances=[{"content": text}],
#                 parameters={"task": "token_count"},
#             )
#             token_count = response.predictions[0]["token_count"]
#             return token_count

#         if issubclass(cls, FixedTokenChunker):
#             extra_kwargs = {
#                 "model_name": model_name,
#             }
#             kwargs = {**kwargs, **extra_kwargs}

#         return cls(length_function=_gemini_tokenizer, **kwargs)
#     def from_gemma_tokenizer(
#         cls: Type[TS],
#         model_path: str,
#         **kwargs: Any,
#     ) -> TS:
#         """Create a TextSplitter that uses Gemma tokenizer for token counting."""
#         tokenizer = GemmaTokenizer(model_path=model_path)

#         def _gemma_tokenizer(text: str) -> int:
#             return len(tokenizer.encode(text))

#         if issubclass(cls, FixedTokenChunker):
#             extra_kwargs = {
#                 "model_path": model_path,
#             }
#             kwargs = {**kwargs, **extra_kwargs}

#         return cls(length_function=_gemma_tokenizer, **kwargs)

# class FixedTokenChunker(TextSplitter):
#     """Splitting text to tokens using either tiktoken or Gemini tokenizers."""

#     def __init__(
#         self,
#         encoding_name: Optional[str] = None,
#         model_name: Optional[str] = None,
#         gemma_model_path: Optional[str] = None,
#         chunk_size: int = 2000,
#         chunk_overlap: int = 200,
#         allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
#         disallowed_special: Union[Literal["all"], Collection[str]] = "all",
#         **kwargs: Any,
#     ) -> None:
#         """Create a new FixedTokenChunker."""
#         super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=kwargs.get('length_function', len))

#         self.encoding_name = encoding_name
#         self.model_name = model_name
#         self.allowed_special = allowed_special
#         self.disallowed_special = disallowed_special
        
#         self._tokenizer = None
#         self.gemma_tokenizer = None 

#         # Set up tokenizer
#         if gemma_model_path:
#             try:
#                 self.gemma_tokenizer = GemmaTokenizer(model_path=gemma_model_path)
#             except Exception as e:
#                 logger.error(f"Failed to load Gemma tokenizer: {e}")
#         # Set up tokenizer for OpenAI models using tiktoken
#         elif encoding_name:
#             try:
#                 import tiktoken
#                 if model_name:
#                     self._tokenizer = tiktoken.encoding_for_model(model_name)
#                 else:
#                 # Check if encoding_name is supported by tiktoken
#                     try:
#                         self._tokenizer = tiktoken.get_encoding(encoding_name)
#                     except ValueError:
#                 # If encoding_name is not recognized by tiktoken, set _tokenizer to None
#                         logger.warning(f"Unknown encoding name '{encoding_name}'. Falling back to Gemini tokenizer if available.")
#                         self._tokenizer = None
#             except ImportError:
#                 raise ImportError(
#                     "Could not import tiktoken python package. "
#                     "Please install it with `pip install tiktoken`."
#                 )

# # Set up Gemini tokenizer if the model name indicates a Gemini model or tiktoken failed
#         elif model_name and (encoding_name is None or self._tokenizer is None):
#             try:
#                 from google.cloud import aiplatform
#                 self.client = aiplatform.gapic.ModelServiceClient()
#                 self.model_name = model_name
#             except ImportError:
#                 raise ImportError(
#                     "Could not import Google Cloud AI Platform. "
#                     "Ensure that the Google Cloud SDK is installed properly."
#                 )


#     def split_text(self, text: str) -> List[str]:
#         def _encode(_text: str) -> List[int]:
#             # If tiktoken tokenizer is available, use it
#             if self._tokenizer is not None:
#                 return self._tokenizer.encode(
#                     _text,
#                     allowed_special=self.allowed_special,
#                     disallowed_special=self.disallowed_special,
#                 )
#             # Otherwise, use Gemini tokenizer if available
#             elif hasattr(self, 'client') and self.client is not None:
#                 response = self.client.predict(
#                     name=self.model_name,
#                     instances=[{"content": _text}],
#                     parameters={"task": "token_count"},
#                 )
#                 token_count = response.predictions[0]["token_count"]
#                 return list(range(token_count))
#             # Otherwise, use Gemma tokenizer if available
#             elif self.gemma_tokenizer is not None:
#                 return self.gemma_tokenizer.encode(_text)
#             else:
#                 raise ValueError("No valid tokenizer found for encoding.")
#         def _decode(tokens: List[int]) -> str:
#             # If tiktoken tokenizer is available, use it
#             if self._tokenizer is not None:
#                 return self._tokenizer.decode(tokens)
#             # If Gemma tokenizer is available, use it
#             elif self.gemma_tokenizer is not None:
#                 return self.gemma_tokenizer.decode(tokens)
#             else:
#                 raise ValueError("No valid tokenizer found for decoding.")

#         tokenizer = Tokenizer(
#             chunk_overlap=self._chunk_overlap,
#             tokens_per_chunk=self._chunk_size,
#             decode=_decode,
#             encode=_encode,
#         )

#         return split_text_on_tokens(text=text, tokenizer=tokenizer)

# @dataclass(frozen=True)
# class Tokenizer:
#     """Tokenizer data class."""

#     chunk_overlap: int
#     tokens_per_chunk: int
#     decode: Callable[[List[int]], str]
#     encode: Callable[[str], List[int]]

# def split_text_on_tokens(*, text: str, tokenizer: Tokenizer) -> List[str]:
#     """Split incoming text and return chunks using tokenizer."""
#     splits: List[str] = []
#     input_ids = tokenizer.encode(text)
#     start_idx = 0
#     cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
#     chunk_ids = input_ids[start_idx:cur_idx]
#     while start_idx < len(input_ids):
#         splits.append(tokenizer.decode(chunk_ids))
#         if cur_idx == len(input_ids):
#             break
#         start_idx += tokenizer.tokens_per_chunk - tokenizer.chunk_overlap
#         cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
#         chunk_ids = input_ids[start_idx:cur_idx]
#     return splits

from abc import ABC, abstractmethod
import logging
from typing import (
    AbstractSet,
    Any,
    Callable,
    Collection,
    Iterable,
    List,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
)
from .base_chunker import BaseChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings  
from langchain_google_genai import ChatGoogleGenerativeAI
from google.cloud import aiplatform 
from attr import dataclass
import sentencepiece
import os
from attr import dataclass

logger = logging.getLogger(__name__)

TS = TypeVar("TS", bound="TextSplitter")
class GemmaTokenizer:
    """Tokenizer class to use Gemma Tokenizer with SentencePiece."""

    def __init__(self, model_path: Optional[str]):
        # Load tokenizer model from file
        assert os.path.isfile(model_path), f"Model file not found: {model_path}"
        self.sp_model = sentencepiece.SentencePieceProcessor()
        self.sp_model.Load(model_path)

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.GetPieceSize()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()

    def encode(self, s: str, bos: bool = True, eos: bool = False) -> List[int]:
        """Converts a string into a list of tokens."""
        assert isinstance(s, str)
        tokens = self.sp_model.EncodeAsIds(s)
        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Converts a list of tokens into a string."""
        return self.sp_model.DecodeIds(tokens)
class TextSplitter(BaseChunker, ABC):
    """Interface for splitting text into chunks."""

    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        keep_separator: bool = False,
        add_start_index: bool = False,
        strip_whitespace: bool = True,
    ) -> None:
        """Create a new TextSplitter."""
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._keep_separator = keep_separator
        self._add_start_index = add_start_index
        self._strip_whitespace = strip_whitespace

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into multiple components."""

    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        text = separator.join(docs)
        if self._strip_whitespace:
            text = text.strip()
        if text == "":
            return None
        else:
            return text

    def _merge_splits(self, splits: Iterable[str], separator: str) -> List[str]:
        # Merges smaller pieces into medium-sized chunks.
        separator_len = self._length_function(separator)
        docs = []
        current_doc: List[str] = []
        total = 0
        for d in splits:
            _len = self._length_function(d)
            if (
                total + _len + (separator_len if len(current_doc) > 0 else 0)
                > self._chunk_size
            ):
                if total > self._chunk_size:
                    logger.warning(
                        f"Created a chunk of size {total}, "
                        f"which is longer than the specified {self._chunk_size}"
                    )
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    while total > self._chunk_overlap or (
                        total + _len + (separator_len if len(current_doc) > 0 else 0)
                        > self._chunk_size
                        and total > 0
                    ):
                        total -= self._length_function(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs

    @classmethod
    def from_tiktoken_encoder(
        cls: Type[TS],
        encoding_name: str = "gpt2",
        model_name: Optional[str] = None,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = "all",
        **kwargs: Any,
    ) -> TS:
        """Create a TextSplitter that uses tiktoken encoder for token counting."""
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "Could not import tiktoken python package. "
                "Please install it with pip install tiktoken."
            )

        if model_name is not None:
            enc = tiktoken.encoding_for_model(model_name)
        else:
            enc = tiktoken.get_encoding(encoding_name)

        def _tiktoken_encoder(text: str) -> int:
            return len(
                enc.encode(
                    text,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        if issubclass(cls, FixedTokenChunker):
            extra_kwargs = {
                "encoding_name": encoding_name,
                "model_name": model_name,
                "allowed_special": allowed_special,
                "disallowed_special": disallowed_special,
            }
            kwargs = {**kwargs, **extra_kwargs}

        return cls(length_function=_tiktoken_encoder, **kwargs)

    def from_gemini_tokenizer(
        cls: Type[TS],
        model_name: str = "gemini-1.5-flash-002",
        **kwargs: Any,
    ) -> TS:
        """Create a TextSplitter that uses Gemini tokenizer for token counting."""
        client = aiplatform.gapic.ModelServiceClient()

        def _gemini_tokenizer(text: str) -> int:
            response = client.predict(
                name=model_name,
                instances=[{"content": text}],
                parameters={"task": "token_count"},
            )
            token_count = response.predictions[0]["token_count"]
            return token_count

        if issubclass(cls, FixedTokenChunker):
            extra_kwargs = {
                "model_name": model_name,
            }
            kwargs = {**kwargs, **extra_kwargs}

        return cls(length_function=_gemini_tokenizer, **kwargs)
    def from_gemma_tokenizer(
        cls: Type[TS],
        model_path: str,
        **kwargs: Any,
    ) -> TS:
        """Create a TextSplitter that uses Gemma tokenizer for token counting."""
        tokenizer = GemmaTokenizer(model_path=model_path)

        def _gemma_tokenizer(text: str) -> int:
            return len(tokenizer.encode(text))

        if issubclass(cls, FixedTokenChunker):
            extra_kwargs = {
                "model_path": model_path,
            }
            kwargs = {**kwargs, **extra_kwargs}

        return cls(length_function=_gemma_tokenizer, **kwargs)

class FixedTokenChunker(TextSplitter):
    """Splitting text to tokens using either tiktoken or Gemini tokenizers."""

    def __init__(
        self,
        encoding_name: Optional[str] = None,
        model_name: Optional[str] = None,
        gemma_model_path: Optional[str] = None,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = "all",
        **kwargs: Any,
    ) -> None:
        """Create a new FixedTokenChunker."""
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=kwargs.get('length_function', len))

        self.encoding_name = encoding_name
        self.model_name = model_name
        self.allowed_special = allowed_special
        self.disallowed_special = disallowed_special
        
        self._tokenizer = None
        self.gemma_tokenizer = None 

        # Set up tokenizer
        if gemma_model_path:
            try:
                self.gemma_tokenizer = GemmaTokenizer(model_path=gemma_model_path)
            except Exception as e:
                logger.error(f"Failed to load Gemma tokenizer: {e}")
        # Set up tokenizer for OpenAI models using tiktoken
        elif encoding_name:
            try:
                import tiktoken
                if model_name:
                    self._tokenizer = tiktoken.encoding_for_model(model_name)
                else:
                # Check if encoding_name is supported by tiktoken
                    try:
                        self._tokenizer = tiktoken.get_encoding(encoding_name)
                    except ValueError:
                # If encoding_name is not recognized by tiktoken, set _tokenizer to None
                        logger.warning(f"Unknown encoding name '{encoding_name}'. Falling back to Gemini tokenizer if available.")
                        self._tokenizer = None
            except ImportError:
                raise ImportError(
                    "Could not import tiktoken python package. "
                    "Please install it with pip install tiktoken."
                )

# Set up Gemini tokenizer if the model name indicates a Gemini model or tiktoken failed
        elif model_name and (encoding_name is None or self._tokenizer is None):
            try:
                from google.cloud import aiplatform
                self.client = aiplatform.gapic.ModelServiceClient()
                self.model_name = model_name
            except ImportError:
                raise ImportError(
                    "Could not import Google Cloud AI Platform. "
                    "Ensure that the Google Cloud SDK is installed properly."
                )


    def split_text(self, text: str) -> List[str]:
        def _encode(_text: str) -> List[int]:
            # If tiktoken tokenizer is available, use it
            if self._tokenizer is not None:
                return self._tokenizer.encode(
                    _text,
                    allowed_special=self.allowed_special,
                    disallowed_special=self.disallowed_special,
                )
            # Otherwise, use Gemini tokenizer if available
            elif hasattr(self, 'client') and self.client is not None:
                response = self.client.predict(
                    name=self.model_name,
                    instances=[{"content": _text}],
                    parameters={"task": "token_count"},
                )
                token_count = response.predictions[0]["token_count"]
                return list(range(token_count))
            # Otherwise, use Gemma tokenizer if available
            elif self.gemma_tokenizer is not None:
                return self.gemma_tokenizer.encode(_text)
            else:
                raise ValueError("No valid tokenizer found for encoding.")
        def _decode(tokens: List[int]) -> str:
            # If tiktoken tokenizer is available, use it
            if self._tokenizer is not None:
                return self._tokenizer.decode(tokens)
            # If Gemma tokenizer is available, use it
            elif self.gemma_tokenizer is not None:
                return self.gemma_tokenizer.decode(tokens)
            else:
                raise ValueError("No valid tokenizer found for decoding.")

        tokenizer = Tokenizer(
            chunk_overlap=self._chunk_overlap,
            tokens_per_chunk=self._chunk_size,
            decode=_decode,
            encode=_encode,
        )

        return split_text_on_tokens(text=text, tokenizer=tokenizer)

@dataclass(frozen=True)
class Tokenizer:
    """Tokenizer data class."""

    chunk_overlap: int
    tokens_per_chunk: int
    decode: Callable[[List[int]], str]
    encode: Callable[[str], List[int]]

def split_text_on_tokens(*, text: str, tokenizer: Tokenizer) -> List[str]:
    """Split incoming text and return chunks using tokenizer."""
    splits: List[str] = []
    input_ids = tokenizer.encode(text)
    start_idx = 0
    cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
    chunk_ids = input_ids[start_idx:cur_idx]
    while start_idx < len(input_ids):
        splits.append(tokenizer.decode(chunk_ids))
        if cur_idx == len(input_ids):
            break
        start_idx += tokenizer.tokens_per_chunk - tokenizer.chunk_overlap
        cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]
    return splits
