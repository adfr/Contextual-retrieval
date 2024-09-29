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
        chunks.append("Summary: " + summary + "\n" + current_chunk.strip())
    
    return chunks

def generate_summary(text):
    openai.api_key = openai_api_key
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text."},
            {"role": "user", "content": f"Please summarize the following text in a concise manner:\n\n{text}"}
        ],
        max_tokens=150
    )
    return response.choices[0].message['content'].strip()

def process_wikipedia_page(topic,summary_bool=True):
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
            wiki_collection.add(
                documents=[chunk],
                metadatas=[{"title": title, "chunk_id": i}],
                ids=[f"{title.replace(' ', '_')}_{i}"]
            )
        except Exception as e:
            print(f"Error adding chunk {i} to Chroma: {e}")
def query_rag_system(query, num_results=5):
    # Search for relevant documents
    results = wiki_collection.query(
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
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the given context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
        ],
        max_tokens=300
    )

    return response.choices[0].message['content'].strip()

# Example usage

# Example usage
topics = [
    "Artificial Intelligence", "Machine Learning", "Natural Language Processing",
    "Deep Learning", "Neural Networks", "Computer Vision", "Robotics",
    "Expert Systems", "Genetic Algorithms", "Fuzzy Logic", "Knowledge Representation",
    "Automated Reasoning", "Planning and Decision Making", "Speech Recognition",
    "Natural Language Generation", "Sentiment Analysis", "Machine Translation",
    "Information Retrieval", "Text Mining", "Data Mining", "Big Data Analytics",
    "Predictive Analytics", "Prescriptive Analytics", "Business Intelligence",
    "Internet of Things", "Cloud Computing", "Edge Computing", "Quantum Computing",
    "Blockchain", "Cybersecurity", "Cryptography", "Virtual Reality",
    "Augmented Reality", "Mixed Reality", "3D Printing", "Nanotechnology",
    "Biotechnology", "Cognitive Science", "Neuroscience", "Human-Computer Interaction",
    "User Experience Design", "Autonomous Vehicles", "Drones", "Smart Cities",
    "Smart Homes", "Wearable Technology", "Biometrics", "Facial Recognition",
    "Voice Recognition", "Gesture Recognition", "Emotion Recognition",
    "Recommender Systems", "Personalization", "Digital Twin", "Simulation",
    "Augmented Analytics", "AutoML", "Transfer Learning", "Reinforcement Learning",
    "Unsupervised Learning", "Semi-supervised Learning", "Federated Learning",
    "Explainable AI", "Ethical AI", "AI Governance", "AI Bias", "AI Safety",
    "AI for Social Good", "Green AI", "Quantum Machine Learning",
    "Neuromorphic Computing", "Swarm Intelligence", "Evolutionary Computation",
    "Computational Creativity", "Affective Computing", "Cognitive Computing",
    "Edge AI", "Embedded AI", "AI Chips", "AI-powered IoT", "5G and AI",
    "AI in Healthcare", "AI in Finance", "AI in Education", "AI in Agriculture",
    "AI in Manufacturing", "AI in Retail", "AI in Energy", "AI in Transportation",
    "AI in Space Exploration", "AI in Environmental Science", "AI in Law",
    "AI in Marketing", "AI in Customer Service", "AI in Human Resources",
    "AI in Gaming", "AI in Art and Music", "AI Ethics and Philosophy",
    "AI Policy and Regulation", "AI Economics", "AI and Climate Change"
]
for topic in tqdm(topics):
    process_wikipedia_page(topic,True)

print("RAG system built successfully!")

# Example queries
example_queries = [
    "What is the Turing test?",
    "Is linear optimization part of AI?"
]

print("\nExample queries:")
for query in example_queries:
    print(f"\nQuestion: {query}")
    answer = query_rag_system(query)
    
    print(f"Answer: {answer}")
