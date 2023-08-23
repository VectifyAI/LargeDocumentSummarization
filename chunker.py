import tiktoken
import math


def get_token_size(document, model):
    tokenizer=tiktoken.encoding_for_model(model)
    return len(tokenizer.encode(document))

def naive_chunker(document, chunk_size, model):
    tokenizer=tiktoken.encoding_for_model(model)
    document_tokens=tokenizer.encode(document)
    document_size = len(document_tokens)
    
    chunks = []
    for i in range(0, document_size, chunk_size):
        chunk = document_tokens[i:i + chunk_size]
        chunks.append(tokenizer.decode(chunk))

    return chunks


def auto_chunker(document, max_chunk_size, model):
    tokenizer = tiktoken.encoding_for_model(model)
    document_tokens = tokenizer.encode(document)
    document_size = len(document_tokens)
    # total chunk number
    K = math.ceil(document_size / max_chunk_size)
    # average integer chunk size
    average_chunk_size = math.ceil(document_size / K)
    # number of chunks with average_chunk_size - 1 
    shorter_chunk_number = K * average_chunk_size - document_size
    # number of chunks with average_chunk_size
    standard_chunk_number = K - shorter_chunk_number

    chunks = []
    chunk_start = 0
    for i in range(0, K):
        if i < standard_chunk_number:
            chunk_end = chunk_start + average_chunk_size
        else:
            chunk_end = chunk_start + average_chunk_size - 1
        chunk = document_tokens[chunk_start:chunk_end]
        chunks.append(tokenizer.decode(chunk))
        chunk_start = chunk_end

    assert chunk_start == document_size
    return chunks