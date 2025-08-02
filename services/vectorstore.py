import google.generativeai as genai
import time
from config import GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX, EMBEDDING_MODEL, EMBEDDING_DIMENSION
from pinecone import Pinecone, ServerlessSpec

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

def create_temp_index() -> str:
    # print(f"Starting create_temp_index()")
    # print(f"PINECONE_INDEX: {PINECONE_INDEX}")
    # print(f"PINECONE_ENV: {PINECONE_ENV}")
    
    index_name = PINECONE_INDEX
    # print(f"Using index_name: {index_name}")
    
    try:
        # print(f"Checking if index exists: {index_name}")
        has_index = pc.has_index(index_name)
        # print(f"Index exists: {has_index}")
        
        if not has_index:
            # print(f"Creating new index: {index_name}")
            # print(f"Using region: {PINECONE_ENV}")
            # print(f"Using dimension: {EMBEDDING_DIMENSION}")
            
            pc.create_index(
                name=index_name,
                dimension=EMBEDDING_DIMENSION,  # 768 for Gemini
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
            )
            # print(f"Index created successfully: {index_name}")
        else:
            print(f"Index already exists: {index_name}")
        
        print(f"create_temp_index() returning: {index_name}")
        return index_name
        
    except Exception as e:
        print(f"Error in create_temp_index(): {e}")
        raise e

def embed_chunks_store(index_name: str, chunks: list[str]):
    print(f"Starting to embed and store {len(chunks)} chunks")
    index = pc.Index(index_name)
    
    # Process one chunk at a time to respect rate limits
    vectors = []
    
    for i, chunk in enumerate(chunks):
        try:
            embedding = get_embedding(chunk)
            vectors.append({
                "id": f"chunk-{i}",
                "values": embedding,
                "metadata": {"text": chunk}
            })
            
        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")
            if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                print("Rate limit hit, waiting 60 seconds...")
                time.sleep(60)
                # Retry this chunk
                try:
                    embedding = get_embedding(chunk)
                    vectors.append({
                        "id": f"chunk-{i}",
                        "values": embedding,
                        "metadata": {"text": chunk}
                    })
                    print(f"Chunk {i+1} processed successfully (retry)")
                except Exception as retry_error:
                    print(f"Chunk {i+1} failed even after retry: {retry_error}")
                    raise retry_error
            else:
                raise e
    
    # Upload to Pinecone in batches
    upload_batch_size = 100
    
    for i in range(0, len(vectors), upload_batch_size):
        batch_vectors = vectors[i:i + upload_batch_size]
        try:
            index.upsert(vectors=batch_vectors)
        except Exception as e:
            print(f"Upload batch {i//upload_batch_size + 1} failed: {e}")
            raise e
    
    print(f"Successfully stored {len(chunks)} chunks in index: {index_name}")

def get_embedding(text: str) -> list[float]:
    """Get embedding for a single text using Gemini API."""
    try:
        # Truncate text if too long (Gemini has input limits)
        if len(text) > 2048:
            text = text[:2048]
            print(f"Text truncated to 2048 characters")
        
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document"
        )
        
        embedding = result['embedding']
        return embedding
        
    except Exception as e:
        print(f"Gemini embedding failed: {e}")
        print(f"Text sample: {text[:100]}...")
        
        # Handle common errors
        if "rate limit" in str(e).lower() or "quota" in str(e).lower():
            print("Rate limit hit, waiting before retry...")
            time.sleep(60)  # Wait 1 minute
            # Retry once
            try:
                result = genai.embed_content(
                    model=EMBEDDING_MODEL,
                    content=text,
                    task_type="retrieval_document"
                )
                return result['embedding']
            except Exception as retry_error:
                raise Exception(f"Failed to get embedding after retry: {retry_error}")
        else:
            raise Exception(f"Failed to get embedding: {e}")

def get_query_embedding(text: str) -> list[float]:
    """Get embedding for query text."""
    try:
        print(f"Getting Gemini query embedding for: {text[:50]}...")
        
        # Truncate if too long
        if len(text) > 2048:
            text = text[:2048]
        
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_query"  # Different task type for queries
        )
        
        embedding = result['embedding']
        print(f"Got query embedding vector of length: {len(embedding)}")
        return embedding
        
    except Exception as e:
        print(f"Query embedding failed: {e}")
        raise Exception(f"Failed to get query embedding: {e}")

def test_gemini_connection():
    """Test Gemini API connection."""
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content="test connection",
            task_type="retrieval_document"
        )
        print("Gemini API connection successful!")
        print(f"Embedding dimension: {len(result['embedding'])}")
        return True
    except Exception as e:
        print(f"Gemini API connection failed: {e}")
        return False