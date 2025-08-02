import google.generativeai as genai
from config import GEMINI_API_KEY, PINECONE_API_KEY
from pinecone import Pinecone
from services.vectorstore import get_query_embedding
import asyncio

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

TOP_K = 5

async def process_single_query(question: str, index_name: str, namespace: str = "") -> str:
    RELEVANCE_THRESHOLD = 0.75
    DEBUG_MATCH_SCORES = False

    try:
        print(f"Processing query: {question[:50]}...")
        
        # Get query embedding using Gemini
        query_embed = await asyncio.to_thread(get_query_embedding, question)

        # Search in Pinecone
        index = pc.Index(index_name)  
        response = index.query(
            vector=query_embed,
            top_k=TOP_K,
            include_metadata=True,
            namespace=namespace
        )

        matches = response.get("matches", [])
        print(f"Found {len(matches)} matches")
        
        if DEBUG_MATCH_SCORES:
            for m in matches:
                print(f"Score: {m['score']:.4f} | Snippet: {m['metadata']['text'][:80]}")

        # Filter relevant matches
        relevant_matches = [m for m in matches if m['score'] >= RELEVANCE_THRESHOLD]
        if not relevant_matches and matches:
            relevant_matches = matches[:1]
            print(f"No highly relevant matches, using best match with score {matches[0]['score']:.4f}")

        if not relevant_matches:
            return "I couldn't find relevant information in the document to answer your question."

        # Build context from relevant matches
        context = "\n\n".join(m["metadata"]["text"] for m in relevant_matches)
        # print(f"Built context from {len(relevant_matches)} matches")
        
        # Generate answer using Gemini
        answer = await generate_answer_with_gemini(context, question)
        return answer

    except Exception as e:
        print(f"Error processing query: {e}")
        return f"Error processing query: {str(e)}"

async def generate_answer_with_gemini(context: str, question: str) -> str:
    """Generate answer using Gemini's text generation."""
    try:
        # print("Generating answer with Gemini...")
        
        prompt = f"""Based on the following context from a document, answer the question clearly and precisely.

Context:
{context}

Question: {question}

Instructions:
- Answer based only on the information provided in the context
- Be specific and precise
- If the context doesn't contain enough information to answer the question, say "The provided document doesn't contain sufficient information to answer this question."
- Quote relevant parts from the context when helpful

Answer:"""

       
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        # Run in thread to make it async
        def _generate():
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=512,
                )
            )
            return response.text
        
        answer = await asyncio.to_thread(_generate)
        # print(f"Generated answer: {len(answer)} characters")
        return answer.strip()
        
    except Exception as e:
        print(f"Gemini answer generation failed: {e}")
        # Fallback to simple answer if Gemini fails
        return generate_simple_answer(context, question)

def generate_simple_answer(context: str, question: str) -> str:
    """Fallback simple answer generation."""
    question_lower = question.lower()
    context_sentences = context.split('.')
    
    # Find most relevant sentences
    relevant_sentences = []
    keywords = question_lower.split()
    
    for sentence in context_sentences:
        sentence_lower = sentence.lower()
        score = sum(1 for keyword in keywords if keyword in sentence_lower)
        if score > 0:
            relevant_sentences.append((sentence.strip(), score))
    
    # Sort by relevance and take top sentences
    relevant_sentences.sort(key=lambda x: x[1], reverse=True)
    top_sentences = [sent[0] for sent in relevant_sentences[:3] if sent[0]]
    
    if top_sentences:
        answer = "Based on the document:\n\n" + "\n\n".join(top_sentences)
        return answer
    else:
        return "I found some relevant information but couldn't extract a specific answer to your question."
