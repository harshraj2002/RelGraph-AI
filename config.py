"""
Configuration setting for Neo4j RAG Chatbot
"""

class Config:
    #Neo4j Connection Setting
    NEO4J_URI = "neo4j://127.0.0.1:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "Harsh@123"
    NEO4J_DATABASE = "testdb"
    
    #Connection management setting
    CONNECTION_TIMEOUT = 60
    MAX_RETRIES = 3
    RETRY_DELAY = 2
    
    #Model Configuration for embedding and language model
    EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1"
    LLM_MODEL = "llama3.2"
    
    #RAG System Settings for retrieval and generation
    MAX_CONTEXT_LENGTH = 3000  
    TOP_K_RESULTS = 8  
    SIMILARITY_THRESHOLD = 0.1
    
    #Knowledge Graph Structure Setting
    DATASET_LABELS = ["Dataset1", "Dataset2", "Dataset3", "Dataset4"]
    ENTITY_LABELS = ["User", "Analyst", "Contributor", "Issue", "Trend", "Idea", "Platform"]
    PLATFORM_NAMES = ["InsightUX", "Exsight", "Afkari", "InnovateX"]
    
    #Triplet Supervisor Setting
    DEFAULT_TRIPLET_BATCH_SIZE = 50
    TRIPLET_VALIDATION_SAMPLE_SIZE = 100
    SYNTHETIC_DATA_DEFAULT_COUNT = 300
    
    #Processing settings for batch operation
    EMBEDDING_BATCH_SIZE = 50