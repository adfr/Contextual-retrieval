import chromadb
from chromadb.utils import embedding_functions
import openai
import wikipedia
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
import os
from dotenv import load_dotenv
# Initialize ChromaDB client
chroma_client = chromadb.Client()
import os

# Get the OpenAI API key from the environment variable

load_dotenv()
openai_api_key= os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")


# Initialize OpenAI embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_api_key,
    model_name="text-embedding-ada-002"
)

# Create a collection for Wikipedia pages
wiki_collection = chroma_client.create_collection(name="wikipedia_rag", embedding_function=openai_ef)
wiki_collection_summary = chroma_client.create_collection(name="wikipedia_rag_summary", embedding_function=openai_ef)

def get_wikipedia_content(topic):
    try:
        page = wikipedia.page(topic)
        return page.title, page.content
    except wikipedia.exceptions.DisambiguationError as e:
        return e.options[0], wikipedia.page(e.options[0]).content
    except wikipedia.exceptions.PageError:
        return None, None

def chunk_text(text, summary=None, chunk_size=5000, overlap=200):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        if summary:
            chunks.append("Summary: " + summary + "\n" + current_chunk.strip())
        else:
            chunks.append(current_chunk.strip())
    else:
        chunks.append(current_chunk.strip())
    
    return chunks

def generate_summary(text):
    openai.api_key = openai_api_key
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text.  If the context is not relevant, please answer with 'I don't know'."},
            {"role": "user", "content": f"Please summarize the following text in a concise manner:\n\n{text}"}
        ],
        max_tokens=150
    )
    return response.choices[0].message['content'].strip()

def process_wikipedia_page(topic,index_use, summary_bool=True):
    title, content = get_wikipedia_content(topic)
    if not content:
        print(f"Could not fetch content for {topic}")
        return
    if summary_bool:
        summary = generate_summary(content)
        chunks = chunk_text(content,summary)
    else:
        summary = None
        chunks = chunk_text(content)
    
    
    for i, chunk in enumerate(chunks):
        try:
            index_use.add(
                documents=[chunk],
                metadatas=[{"title": title, "chunk_id": i}],
                ids=[f"{title.replace(' ', '_')}_{i}"]
            )
        except Exception as e:
            print(f"Error adding chunk {i} to Chroma: {e}")
def query_rag_system(query, index_use,  num_results=5):
    # Search for relevant documents
    results = index_use.query(
        query_texts=[query],
        n_results=num_results
    )

    # Combine the retrieved chunks
    print(results)
    context = "\n".join(results['documents'][0])
    

    # Generate an answer using OpenAI's API
    openai.api_key = openai_api_key
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the given context. If the context is not relevant, please answer with 'I don't know'."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
        ],
        max_tokens=300
    )

    return response.choices[0].message['content'].strip()
def retrieve_chunks(query, index_full, index_summary, num_results=5):
    # Retrieve chunks from both indexes
    results_full = index_full.query(
        query_texts=[query],
        n_results=num_results
    )
    results_summary = index_summary.query(
        query_texts=[query],
        n_results=num_results
    )

    # Check if the retrieved chunks match
    full_ids = set([doc.split('_')[0] for doc in results_full['ids'][0]])
    summary_ids = set([doc.split('_')[0] for doc in results_summary['ids'][0]])
    matching = full_ids.intersection(summary_ids)

    print(f"Matching documents: {len(matching)} out of {num_results}")
    print(f"Full index documents: {full_ids}")
    print(f"Summary index documents: {summary_ids}")

    return len(matching)/num_results
# Example usage

# Example usage
topics = [
    "Artificial Intelligence", "Machine Learning", "Natural Language Processing",
    "Deep Learning", "Neural Networks", "Computer Vision"]
''',
    "Robotics",
    "Expert Systems", "Genetic Algorithms", "Fuzzy Logic", "Knowledge Representation"  "Automated Reasoning", "Planning and Decision Making", "Speech Recognition",
    "Natural Language Generation", "Sentiment Analysis", "Machine Translation",
    "Information Retrieval", "Text Mining", "Data Mining", "Big Data Analytics",
    "Predictive Analytics", "Prescriptive Analytics", "Business Intelligence",
    "Internet of Things", "Cloud Computing", "Edge Computing", "Quantum Computing",
    "Blockchain", "Cybersecurity", "Cryptography", "Virtual Reality" '''

for topic in tqdm(topics):
    process_wikipedia_page(topic,wiki_collection_summary,True)
    process_wikipedia_page(topic,wiki_collection,False)

print("RAG system built successfully!")

# Example queries
# Load queries from querysample.csv
with open('querysample.csv', 'r') as file:
    example_queries = [line.strip() for line in file if line.strip()]


print("\nExample queries:")

from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
import nltk
import json
from datetime import datetime

nltk.download('punkt', quiet=True)

def evaluate_responses(response1, response2):
    rouge = Rouge()
    rouge_scores = rouge.get_scores(response1, response2)[0]
    
    response1_tokens = nltk.word_tokenize(response1.lower())
    response2_tokens = nltk.word_tokenize(response2.lower())
    bleu_score = sentence_bleu([response1_tokens], response2_tokens)
    
    return {
        "rouge1_f": rouge_scores['rouge-1']['f'],
        "rouge2_f": rouge_scores['rouge-2']['f'],
        "rougeL_f": rouge_scores['rouge-l']['f'],
        "bleu": bleu_score
    }

# Generate responses
responses = {}
total_queries = len(example_queries)
total_matching = 0
matching_counts = {}
generate_response=False
for query in example_queries:
    matching = retrieve_chunks(query, wiki_collection, wiki_collection_summary)

    responses[query] = {
        "with_context": query_rag_system(query, wiki_collection),
        "without_context": query_rag_system(query, wiki_collection_summary),
        "matching_chunks": matching
    }
    
    # Statistics on matching
    num_matching = matching
    total_matching += num_matching
    matching_counts[num_matching] = matching_counts.get(num_matching, 0) + 1
# Print matching counts
print("\nMatching counts:")
for count, frequency in sorted(matching_counts.items()):
    print(f"{count} matching chunk(s): {frequency} queries")

# Calculate and print statistics
average_matching = total_matching / total_queries
print(f"\nMatching statistics:")
print(f"Total queries: {total_queries}")
print(f"Average matching chunks per query: {average_matching:.2f}")
print("Distribution of matching chunks:")
for count, frequency in sorted(matching_counts.items()):
    percentage = (frequency / total_queries) * 100
    print(f"  {count} matching chunk(s): {frequency} queries ({percentage:.2f}%)")

# Save responses
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f'./data/responses_{timestamp}.json', 'w') as f:
    json.dump(responses, f, indent=2)

# Evaluate and save metrics
metrics = {}
for query, answers in responses.items():
    metrics[query] = evaluate_responses(answers["with_context"], answers["without_context"])

with open(f'./data/metrics_{timestamp}.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Print results
print("\nResponses and metrics have been saved.")
print(f"Responses file: responses_{timestamp}.json")
print(f"Metrics file: metrics_{timestamp}.json")

print("\nSample results:")
for query in example_queries:
    print(f"\nQuestion: {query}")
    print(f"Answer (with context): {responses[query]['with_context']}")
    print(f"Answer (without context): {responses[query]['without_context']}")
    print("Evaluation metrics:")
    for metric, value in metrics[query].items():
        print(f"{metric}: {value:.4f}")


