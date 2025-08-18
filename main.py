"""
RelGraph AI - Knowledge Graph Intelligence System
Simplified version focusing only on relationship prediction training and chatbot functionality
"""

import time
import logging
import warnings
import os
from typing import List, Dict, Any, Optional, Tuple
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field
from config import Config
from spo_training_manager import RelationshipPredictionManager

#Suppress all transformer and model loading log for clean console output
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("transformers_modules").setLevel(logging.CRITICAL)
logging.getLogger("sentence_transformers").setLevel(logging.CRITICAL)
logging.getLogger("torch").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class NomicEmbeddings(Embeddings):
    """
    Custom wrapper for Nomic Embed model that handle text embedding
    This provide semantic similarity matching for knowledge graph query
    """
    def __init__(self):
        #Initialize sentence transformer model with trust_remote_code for Nomic model
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL, trust_remote_code=True)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Convert list of text document into vector embedding for similarity search"""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Convert single query text into vector embedding for matching against document"""
        embedding = self.model.encode([text])
        return embedding.tolist()

class GraphRetriever(BaseRetriever):
    """
    Custom Neo4j graph retriever that search knowledge graph using multiple search strategy
    This component find relevant information from the graph database for user query
    """
    driver: Any = Field(description="Neo4j database connection driver")
    embeddings: Any = Field(description="Text embedding model for similarity matching")
    database: str = Field(description="Name of Neo4j database to query")
    k: int = Field(default=8, description="Maximum number of document to retrieve")
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """
        Main retrieval method that search graph using multiple strategy
        Combine entity relationship, link traversal, and connection path for result
        """
        try:
            with self.driver.session(database=self.database) as session:
                documents = []
                
                #Strategy 1: Search for specific entity relationship pattern
                entity_docs = self._search_entity_relationships(session, query)
                documents.extend(entity_docs)
                
                #Strategy 2: Find direct relationship link between entities
                link_docs = self._search_relationship_links(session, query)
                documents.extend(link_docs)
                
                #Strategy 3: Discover multi-hop connection path through the graph
                path_docs = self._search_connection_paths(session, query)
                documents.extend(path_docs)
                
                #Remove duplicate result and return top k document
                unique_docs = self._remove_duplicate_documents(documents)
                return unique_docs[:self.k]
                
        except Exception:
            #Return empty list if any database errors occur to prevent system crash
            return []
    
    def _search_entity_relationships(self, session, query):
        """
        Search for common entity relationship pattern in the knowledge graph
        Uses predefined Cypher query for User-Issue-Platform and other common pattern
        """
        documents = []
        
        #Define common relationship pattern with their associated keyword and Cypher query
        search_patterns = [
            #Pattern for User-Issue-Platform relationship
            {
                "keywords": ["user", "issue", "platform", "status"],
                "cypher": """
                MATCH (user:User)-[:RAISED]->(issue:Issue)-[:ORIGINATED_FROM]->(platform:Platform)
                OPTIONAL MATCH (issue)-[:HAS_STATUS]->(status:Status)
                OPTIONAL MATCH (issue)-[:BELONGS_TO]->(domain:Domain)
                WHERE ANY(word IN split(toLower($search_query), ' ') 
                         WHERE toLower(user.name) CONTAINS word 
                            OR toLower(issue.name) CONTAINS word 
                            OR toLower(platform.name) CONTAINS word)
                RETURN user.name as user_name, issue.name as issue_name, 
                       platform.name as platform_name, status.name as status_name,
                       domain.name as domain_name,
                       'User ' + user.name + ' raised issue ' + issue.name + 
                       ' on platform ' + platform.name + 
                       ' with status ' + coalesce(status.name, 'Unknown') +
                       ' in domain ' + coalesce(domain.name, 'Unknown') as description
                LIMIT 3
                """,
                "type": "user_issue_platform"
            },
            #Pattern for Analyst-Trend-Region relationship
            {
                "keywords": ["analyst", "trend", "region", "domain"],
                "cypher": """
                MATCH (analyst:Analyst)-[:AUTHORED]->(trend:Trend)
                OPTIONAL MATCH (trend)-[:OBSERVED_IN]->(region:Region)
                OPTIONAL MATCH (trend)-[:BELONGS_TO]->(domain:Domain)
                WHERE ANY(word IN split(toLower($search_query), ' ') 
                         WHERE toLower(analyst.name) CONTAINS word 
                            OR toLower(trend.name) CONTAINS word 
                            OR toLower(region.name) CONTAINS word
                            OR toLower(domain.name) CONTAINS word)
                RETURN analyst.name as analyst_name, trend.name as trend_name,
                       region.name as region_name, domain.name as domain_name,
                       'Analyst ' + analyst.name + ' authored trend ' + trend.name +
                       ' observed in ' + coalesce(region.name, 'Unknown region') +
                       ' for domain ' + coalesce(domain.name, 'Unknown domain') as description
                LIMIT 3
                """,
                "type": "analyst_trend_region"
            },
            #Pattern for Contributor-Idea-Impact relationship
            {
                "keywords": ["idea", "impact", "contributor", "platform"],
                "cypher": """
                MATCH (contributor:Contributor)-[:PROPOSED]->(idea:Idea)
                OPTIONAL MATCH (idea)-[:HAS_IMPACT_ON]->(impact:ImpactArea)
                OPTIONAL MATCH (idea)-[:ORIGINATED_FROM]->(platform:Platform)
                WHERE ANY(word IN split(toLower($search_query), ' ') 
                         WHERE toLower(contributor.name) CONTAINS word 
                            OR toLower(idea.name) CONTAINS word 
                            OR toLower(impact.name) CONTAINS word
                            OR toLower(platform.name) CONTAINS word)
                RETURN contributor.name as contributor_name, idea.name as idea_name,
                       impact.name as impact_name, platform.name as platform_name,
                       'Contributor ' + contributor.name + ' proposed idea ' + idea.name +
                       ' with impact on ' + coalesce(impact.name, 'Unknown area') +
                       ' from platform ' + coalesce(platform.name, 'Unknown platform') as description
                LIMIT 3
                """,
                "type": "idea_impact_analysis"
            }
        ]
        
        #Execute query for pattern that match the user query keyword
        for pattern in search_patterns:
            if any(keyword in query.lower() for keyword in pattern["keywords"]):
                try:
                    result = session.run(pattern["cypher"], search_query=query)
                    for record in result:
                        description = record.get("description", "Relationship found")
                        #Build metadata with all record field except description
                        metadata = {"type": pattern["type"]}
                        for key, value in record.items():
                            if key != "description" and value:
                                metadata[key] = value
                        #Create document with description and metadata for LLM context
                        doc = Document(page_content=description, metadata=metadata)
                        documents.append(doc)
                except Exception:
                    #Skip pattern that fail due to missing node or relationship
                    continue
        
        return documents
    
    def _search_relationship_links(self, session, query):
        """
        Find direct and two-hop relationship link starting from query-matched node
        This discover connection that might not be captured by predefined patterns
        """
        documents = []
        try:
            #Query to find node matching the search term and their outgoing connection
            link_query = """
            MATCH (start_node)
            WHERE ANY(prop IN keys(start_node) 
                     WHERE toString(start_node[prop]) IS NOT NULL 
                       AND toLower(toString(start_node[prop])) CONTAINS toLower($search_term))
            MATCH (start_node)-[r1]->(connected1)
            OPTIONAL MATCH (connected1)-[r2]->(connected2)
            WITH start_node, r1, connected1, r2, connected2,
                 labels(start_node) as start_label,
                 labels(connected1) as connected1_label,
                 labels(connected2) as connected2_label,
                 coalesce(start_node.name, toString(elementId(start_node))) as start_name,
                 coalesce(connected1.name, toString(elementId(connected1))) as connected1_name,
                 coalesce(connected2.name, toString(elementId(connected2))) as connected2_name
            RETURN start_label, start_name, type(r1) as relationship1_type,
                   connected1_label, connected1_name, type(r2) as relationship2_type,
                   connected2_label, connected2_name
            LIMIT 5
            """
            
            result = session.run(link_query, search_term=query)
            for record in result:
                #Extract node and relationship information from query result
                start_label = record["start_label"]
                start_name = record["start_name"]
                rel1_type = record["relationship1_type"]
                conn1_label = record["connected1_label"]
                conn1_name = record["connected1_name"]
                rel2_type = record.get("relationship2_type")
                conn2_label = record.get("connected2_label")
                conn2_name = record.get("connected2_name")
                
                #Build human-readable description of the relationship chain
                link_description = f"{start_label} '{start_name}' -{rel1_type}-> {conn1_label} '{conn1_name}'"
                if rel2_type and conn2_label and conn2_name:
                    link_description += f" -{rel2_type}-> {conn2_label} '{conn2_name}'"
                
                doc = Document(
                    page_content=f"Link Analysis: {link_description}",
                    metadata={"type": "link_traversal"}
                )
                documents.append(doc)
        except Exception:
            #Skip if link traversal fail to prevent system crash
            pass
        
        return documents
    
    def _search_connection_paths(self, session, query):
        """
        Discover multi-hop path between entities related to the query
        Find indirect connection that span 2-3 relationship hops through the graph
        """
        documents = []
        try:
            #Query for variable-length path connecting query-related node
            path_query = """
            MATCH path = (start)-[*1..3]-(end)
            WHERE ANY(prop IN keys(start) 
                     WHERE toString(start[prop]) IS NOT NULL 
                       AND toLower(toString(start[prop])) CONTAINS toLower($search_term))
               OR ANY(prop IN keys(end) 
                     WHERE toString(end[prop]) IS NOT NULL 
                       AND toLower(toString(end[prop])) CONTAINS toLower($search_term))
            WITH path, length(path) as path_length
            WHERE path_length >= 2
            RETURN nodes(path) as path_nodes, 
                   relationships(path) as path_relationships,
                   path_length
            ORDER BY path_length
            LIMIT 3
            """
            
            result = session.run(path_query, search_term=query)
            for record in result:
                path_nodes = record["path_nodes"]
                path_relationships = record["path_relationships"]
                path_length = record["path_length"]
                
                #Build description of the complete path through multiple node
                path_description_parts = []
                for i in range(len(path_nodes)):
                    node = path_nodes[i]
                    node_label = list(node.labels) if node.labels else "Unknown"
                    node_name = node.get("name", str(node.get("id", "Unknown")))
                    
                    if i < len(path_relationships):
                        rel_type = path_relationships[i].type
                        path_description_parts.append(f"{node_label}({node_name})-[{rel_type}]->")
                    else:
                        path_description_parts.append(f"{node_label}({node_name})")
                
                path_description = "".join(path_description_parts)
                doc = Document(
                    page_content=f"Connection Path: {path_description}",
                    metadata={"type": "connection_path", "path_length": path_length}
                )
                documents.append(doc)
        except Exception:
            #Skip if path discovery fail to prevent system crash
            pass
        
        return documents
    
    def _remove_duplicate_documents(self, documents):
        """
        Remove duplicate document based on content similarity
        Keeps only unique information to avoid redundant response in LLM context
        """
        unique_docs = []
        seen_content = set()
        
        for doc in documents:
            #Use first 100 character as uniqueness key to identify duplicates
            content_key = doc.page_content[:100]
            if content_key not in seen_content:
                unique_docs.append(doc)
                seen_content.add(content_key)
        
        return unique_docs

class RelGraphChatbot:
    """
    Main chatbot class that coordinates all system component
    Handles relationship prediction, graph search, LLM generation, and session accuracy tracking
    """
    
    def __init__(self):
        #Initialize core database connection for Neo4j graph access
        self.driver = self._create_driver()
        
        #Set up AI model for semantic search and text generation
        self.embeddings = NomicEmbeddings()
        self.llm = OllamaLLM(model=Config.LLM_MODEL)
        
        #Create intelligent graph retriever for finding relevant context
        self.retriever = GraphRetriever(
            driver=self.driver,
            embeddings=self.embeddings,
            database=Config.NEO4J_DATABASE,
            k=Config.TOP_K_RESULTS
        )
        
        #Initialize relationship prediction manager
        self.relationship_predictor = None
        
        #Define prompt template for generating responses using graph context
        self.prompt_template = ChatPromptTemplate.from_template("""
You are RelGraph AI, an intelligent assistant that analyzes knowledge graphs with deep understanding of relationships between users, analysts, contributors, issues, trends, and ideas across different platforms.

Context from knowledge graph analysis:
{context}

Question: {question}

Answer based on the knowledge graph relationship and pattern:""")
    
    def _create_driver(self):
        """Create and return Neo4j database connection driver using config setting"""
        driver = GraphDatabase.driver(
            Config.NEO4J_URI,
            auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD),
            encrypted=False
        )
        return driver
    
    def initialize_relationship_predictor(self):
        """Initialize relationship prediction system on demand to save memory"""
        if not self.relationship_predictor:
            self.relationship_predictor = RelationshipPredictionManager()
    
    def train_relationship_prediction(self):
        """
        Train the relationship prediction system on SPO triplets from graph and synthetic data
        This builds pattern for predicting relationship between node types
        """
        if not self.relationship_predictor:
            self.initialize_relationship_predictor()
        
        print("Training relationship prediction system...")
        #Train using extract + generate + pattern building approach
        self.relationship_predictor.train()
        
        return {
            "training_triplets": len(self.relationship_predictor.training_triplets),
            "learned_patterns": len(self.relationship_predictor.predicate_patterns)
        }
    
    def _parse_relationship_question(self, question: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse relationship question to extract subject and object node
        Handle questions like "What is the relationship between X and Y?"
        """
        question_lower = question.lower()
        
        #Try to find "between X and Y" pattern in user question
        if "between" in question_lower and "and" in question_lower:
            try:
                between_idx = question_lower.find("between")
                and_idx = question_lower.find("and", between_idx)
                
                if between_idx != -1 and and_idx != -1:
                    #Extract subject and object from the question
                    subject_part = question[between_idx + 7:and_idx].strip()
                    object_part = question[and_idx + 3:].strip()
                    
                    #Clean up common question ending
                    for ending in ["?", ".", "relationship"]:
                        object_part = object_part.replace(ending, "").strip()
                    
                    return subject_part, object_part
            except:
                #Return None if parsing fail to prevent crash
                pass
        
        return None, None
    
    def predict_node_relationship(self, subject: str, object_name: str, expected: str = None):
        """
        Predict the relationship between two nodes and track for session accuracy
        This is the core method for answering "What is the relationship between X and Y?"
        """
        if not self.relationship_predictor:
            self.initialize_relationship_predictor()
            #Train automatically if not already trained
            self.relationship_predictor.train()
        
        #Use the validate_interaction method which tracks result for accuracy calculation
        result = self.relationship_predictor.validate_interaction(subject, object_name, expected)
        
        return {
            "subject": subject,
            "object": object_name,
            "predicted_relationship": result["predicted"],
            "expected_relationship": result["expected"],
            "correct": result["correct"]
        }
    
    def chat(self, user_query: str) -> str:
        """
        Main conversation method that handles user query with graph-based responses
        Includes special handling for relationship prediction questions with accuracy tracking
        """
        try:
            #Check if user is asking about relationship between specific entities
            if "relationship between" in user_query.lower() or "relation between" in user_query.lower():
                #Parse the question to extract subject and object node name
                subject, obj = self._parse_relationship_question(user_query)
                if subject and obj:
                    #Return direct relationship prediction and track for accuracy
                    prediction = self.predict_node_relationship(subject, obj)
                    return f"The relationship between {subject} and {obj} is: {prediction['predicted_relationship']}"
            
            #For general queries, use graph retrieval and LLM generation
            retrieved_docs = self.retriever._get_relevant_documents(user_query, run_manager=None)
            
            #Build context from retrieved graph information
            context_parts = []
            for doc in retrieved_docs:
                context_parts.append(doc.page_content)
            
            #Prepare context for LLM
            context = "\n".join(context_parts) if context_parts else "No specific information found in knowledge graph."
            
            #Generate response using LLM with graph context
            formatted_prompt = self.prompt_template.format(context=context, question=user_query)
            response = self.llm.invoke(formatted_prompt)
            
            return response
        except Exception:
            #Return user-friendly error message if anything goes wrong
            return "I encountered an error processing your question. Please try rephrasing it."
    
    def get_session_accuracy(self):
        """Get current session accuracy for relationship prediction (0.0 if no predictor)"""
        if not self.relationship_predictor:
            return 0.0
        return self.relationship_predictor.session_accuracy()
    
    def print_session_report(self):
        """Print detailed session accuracy report for all relationship question asked"""
        if not self.relationship_predictor:
            print("No relationship predictor initialized")
            return
        #Delegate to the relationship predictor's reporting method
        self.relationship_predictor.print_session_report()
    
    def close(self):
        """Clean up all database connection and system resources"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
        if self.relationship_predictor:
            self.relationship_predictor.close()

def main():
    """
    Main application entry point with simplified menu
    Only includes relationship prediction training (option 1) and chatbot (option 2)
    """
    chatbot = None
    
    try:
        #Initialize the complete RelGraph AI system
        chatbot = RelGraphChatbot()
        
        #Display system header and available capabilities
        print("RelGraph AI - Knowledge Graph Intelligence System")
        print("=" * 60)
        print("\nSystem Capabilities:")
        print("1. Train relationship prediction and manual testing")
        print("2. Start Chatbot")
        
        #Main menu loop for system capabilities
        while True:
            choice = input("\nSelect option (1, 2) or 'skip' to go directly to chat: ").strip()
            
            if choice == '1':
                #Option 1: Train relationship prediction system
                results = chatbot.train_relationship_prediction()
                print(f"\nRelationship Prediction Training Results:")
                print(f"Training triplets: {results['training_triplets']}")
                print(f"Learned patterns: {results['learned_patterns']}")
                
                #Offer interactive relationship testing after training
                manual_test = input("\nStart interactive relationship testing? (y/n): ").lower()
                if manual_test == 'y':
                    #Ensure predictor is ready for testing
                    if not chatbot.relationship_predictor:
                        chatbot.initialize_relationship_predictor()
                        chatbot.relationship_predictor.train()
                    
                    #Start interactive testing session
                    print("\n" + "="*60)
                    print("MANUAL RELATIONSHIP TESTING")
                    print("="*60)
                    print("Ask questions like: 'What is the relationship between User_1 and Issue_5?'")
                    print("Type 'quit' to return to main menu")
                    print("Type 'accuracy' to see current session accuracy")
                    
                    #Interactive testing loop
                    while True:
                        user_input = input("\nYour question: ").strip()
                        
                        #Handle exit commands
                        if user_input.lower() in ['quit', 'exit', 'q', 'back']:
                            break
                        #Handle accuracy reporting command
                        elif user_input.lower() == 'accuracy':
                            chatbot.print_session_report()
                            continue
                        
                        #Process relationship questions
                        if "relationship between" in user_input.lower() or "relation between" in user_input.lower():
                            response = chatbot.chat(user_input)
                            print(f"Answer: {response}")
                        else:
                            print("Please ask about relationships using format: 'What is the relationship between X and Y?'")
                            
            elif choice == '2' or choice.lower() == 'skip':
                #Option 2: Start chatbot or skip directly to chat
                break
                
            else:
                #Handle invalid menu choices
                print("Invalid option. Please choose 1, 2 or 'skip'")
        
        #Start main chatbot conversation loop
        print("\n" + "="*60)
        print("RelGraph AI is ready for intelligent conversations!")
        print("Ask me about relationships, patterns, and insights from your knowledge graph")
        print("You can ask relationship questions like: 'What is the relationship between User_1 and Issue_5?'")
        print("Type 'accuracy' during chat to see session accuracy")
        print("="*60)
        
        #Main conversation loop
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                #Handle exit command
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("Bot: Thank you for using RelGraph AI! Have a great day!")
                    break
                #Handle accuracy reporting during chat
                elif user_input.lower() == 'accuracy':
                    chatbot.print_session_report()
                    continue
                
                #Skip empty input
                if not user_input:
                    continue
                
                #Generate and display chatbot response
                response = chatbot.chat(user_input)
                print(f"Bot: {response}")
                
            except KeyboardInterrupt:
                #Handle Ctrl+C gracefully
                print("\n\nBot: Session ended. Thank you for using RelGraph AI!")
                break
            except Exception:
                #Handle unexpected errors gracefully
                print("Bot: I'm having trouble with that question. Could you try asking it differently?")
                continue
    
    except Exception as e:
        #Handle system initialization errors
        print(f"System initialization error: {e}")
        print("Please check your configuration and try again.")
    
    finally:
        #Clean up system resources regardless of how the program ends
        if chatbot:
            chatbot.close()
            print("RelGraph AI system shut down successfully.")

if __name__ == "__main__":
    #Run the main application when script is executed directly
    main()