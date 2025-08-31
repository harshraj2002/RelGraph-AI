# RelGraph AI

A Relationship Prediction Knowledge Graph System.

A specialized AI system built on Neo4j knowledge graphs that focuses on training and predicting relationships between entities. RelGraph AI learns from graph patterns and provides accurate relationship predictions with session-based accuracy tracking.


## Overview

RelGraph AI is an intelligent relationship prediction system that leverages Neo4j graph databases to learn patterns between different entity types and predict relationships when asked questions like "What is the relationship between User_1 and Issue_5?" The system combines real graph data with synthetic training examples to achieve high accuracy in relationship prediction.


## Key Features

### Core Functionality
- **Direct Neo4j Integration**: Queries actual relationships from your knowledge graph for real-time accuracy
- **SPO Triplet Learning**: Trains on Subject-Predicate-Object patterns extracted from graph data
- **Hybrid Prediction Strategy**: First checks database for actual relationships, then falls back to learned patterns
- **Session Accuracy Tracking**: Monitors and reports prediction accuracy across conversation sessions

### Training System
- **Real Data Extraction**: Learns from existing relationships in your Neo4j database
- **Synthetic Data Generation**: Creates balanced training examples for comprehensive pattern coverage
- **Pattern Recognition**: Builds predictive models based on node type combinations
- **Interactive Testing**: Manual testing interface for validating relationship predictions

### Intelligence Features
- **Multi-Strategy Retrieval**: Uses entity patterns, link traversal, and connection paths for context
- **Semantic Search**: Powered by Nomic Embed model for similarity matching
- **LLM Integration**: Uses Ollama for natural language generation with graph context
- **Confidence Scoring**: Provides confidence levels for predictions (High/Medium/Low)


## System Architecture

### Core Components

**main.py**
- Main system orchestrator and user interface
- Coordinates relationship prediction, graph search, and conversation management
- Provides interactive menu for training and testing

**spo_training_manager.py**
- Handles SPO (Subject-Predicate-Object) relationship learning and prediction
- Manages session accuracy tracking and detailed reporting
- Implements two-stage prediction: database lookup then pattern matching

**config.py**
- Centralized configuration management
- Database connection settings and model parameters

**neo4j.py**
- Initial knowledge graph setup and population
- Creates foundational entity relationships

### Supported Entity Types
The system works with various entity types including:
- **Users**: People who report issues and propose ideas
- **Analysts**: Experts who create trends and reports
- **Contributors**: People who develop ideas and lead projects
- **Issues**: Problems reported on various platforms
- **Trends**: Analytical findings and observations
- **Ideas**: Proposed solutions and innovations
- **Platforms**: Systems where activities occur (GitHub, Jira, Slack, etc.)
- **Domains**: Subject areas of expertise (Technology, Healthcare, Finance, etc.)
- **Regions**: Geographical areas of operation

### Common Relationship Types
- User RAISED Issue
- User USES Platform  
- Analyst AUTHORED Trend
- Analyst COVERS Region
- Contributor DEVELOPED Idea
- Issue BELONGS_TO Domain
- Trend OBSERVED_IN Region
- Idea HAS_IMPACT_ON ImpactArea


## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Neo4j Database (local or cloud instance)
- Ollama with compatible model (llama2, mistral, or similar)
- Minimum 8GB RAM recommended

### Required Python Packages
```bash
pip install neo4j
pip install sentence-transformers
pip install langchain-core
pip install langchain-ollama
pip install numpy
pip install scikit-learn
pip install pydantic
```

### Neo4j Setup
1. Install and start Neo4j database
2. Create a new database or use existing one
3. Note connection details (URI, username, password)
4. Populate with initial data using `neo4j.py`

### Ollama Setup
1. Install Ollama from official website
2. Pull desired model:
   ```bash
   ollama pull llama2
   ```

### Configuration
Update `config.py` with your specific settings:
```python
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j" 
NEO4J_PASSWORD = "your_password"
NEO4J_DATABASE = "neo4j"
LLM_MODEL = "llama2"
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
TOP_K_RESULTS = 8
```


## Usage Guide

### Starting the System
```bash
python main.py
```

### System Options

**Option 3: Train Relationship Prediction and Manual Testing**
- Extracts SPO triplets from your Neo4j graph
- Generates synthetic training data for comprehensive coverage
- Builds predictive patterns for relationship forecasting
- Offers interactive testing interface for manual validation
- Tracks accuracy in real-time during testing sessions

**Option 5: Start Chatbot**
- Launches intelligent conversation interface
- Handles relationship prediction questions automatically
- Provides graph-based context for general queries
- Supports accuracy tracking across conversation sessions

### Interactive Testing

After training, you can test relationship predictions manually:

```
Your question: What is the relationship between User_1 and Issue_5?
Answer: The relationship between User_1 and Issue_5 is: RAISED

Your question: What is the relationship between Analyst_2 and Trend_3?
Answer: The relationship between Analyst_2 and Trend_3 is: AUTHORED

Your question: accuracy
```

### Sample Queries

**Direct Relationship Questions**
- "What is the relationship between User_1 and Issue_5?"
- "What is the relationship between Analyst_2 and Europe?"
- "What is the relationship between Contributor_3 and Idea_7?"

**General Knowledge Questions**
- "Tell me about User_1's activities"
- "Show me issues in the Technology domain"
- "What trends are observed in North America?"


## Accuracy and Performance

### Prediction Strategy
1. **Database Lookup**: First checks for actual relationship in Neo4j
2. **Pattern Matching**: Falls back to learned patterns if no direct relationship exists
3. **Confidence Scoring**: Provides accuracy confidence (High for database matches, Medium/Low for patterns)

### Session Tracking
- Tracks all relationship questions asked during a session
- Calculates real-time accuracy based on correct predictions
- Provides detailed breakdown by data source (database vs. patterns)
- Shows individual question results with confidence levels

### Performance Metrics
- **Database-verified relationships**: Typically 90%+ accuracy (exact matches)
- **Pattern-based predictions**: 70-85% accuracy (learned from training data)
- **Overall session accuracy**: Varies based on question mix and graph completeness


## Sample Session Report

```
============================================================
RELATIONSHIP PREDICTION SESSION REPORT
============================================================
Total relationship questions asked: 4
Overall accuracy: 100.00%
Database-verified relationships: 2 (Accuracy: 100.00%)
Pattern-based predictions: 2 (Accuracy: 100.00%)

Detailed Results:
1. User_1 -> Issue_5: RAISED [✓ CORRECT] (High confidence)
    Database has: RAISED
2. Project_4 -> Finance: CREATED [✓ CORRECT] (High confidence)
    Database has: CREATED
3. User_50 -> Issue_10: RELATES_TO [✓ CORRECT] (Medium confidence)
    No direct relationship in database
4. Analyst_1 -> Trend_3: AUTHORED [✓ CORRECT] (Medium confidence)
    No direct relationship in database
```


## Technical Implementation

### Training Process
1. **Extract Real Data**: Queries Neo4j for existing SPO triplets
2. **Generate Synthetic Data**: Creates balanced examples using common patterns
3. **Build Patterns**: Learns most common predicates for each (subject_type, object_type) combination
4. **Validate**: Tests predictions against known relationships

### Prediction Logic
```python
def predict_relationship(subject, object):
    # Strategy 1: Check actual database
    direct = check_database(subject, object)
    if direct:
        return direct
    
    # Strategy 2: Use learned patterns
    pattern = get_pattern(subject_type, object_type)
    return pattern or "RELATES_TO"
```

### Graph Search Strategy
- **Entity Relationship Patterns**: Searches for common subgraph patterns
- **Link Traversal**: Follows direct and two-hop relationships
- **Connection Paths**: Discovers multi-hop paths between entities
- **Duplicate Removal**: Ensures unique context for LLM generation


## Development and Customization

### Adding New Entity Types
1. Update entity patterns in `spo_training_manager.py`
2. Add type inference rules in `_infer_type()` method
3. Include in synthetic data generation templates

### Extending Relationship Types
1. Add new relationship patterns to `generate_synthetic_triplets()`
2. Update graph retrieval patterns in `GraphRetriever`
3. Test with new relationship examples

### Customizing Accuracy Thresholds
Modify confidence scoring logic in `track_relationship_question()` method to adjust accuracy calculations.


## Troubleshooting

### Common Issues

**Low Prediction Accuracy**
- Ensure Neo4j database has sufficient relationship data
- Check that entity names follow consistent patterns
- Verify synthetic data generation covers your use cases

**Connection Problems**
- Verify Neo4j is running and accessible
- Check credentials in config.py
- Ensure database exists and contains data

**Performance Issues**
- Reduce training data size for faster processing
- Check available system memory
- Optimize Neo4j database indexes

**Infinite Loop in Reporting**
- System includes protection against infinite loops
- Restart session if accuracy reporting becomes stuck


## File Structure

```
RelGraph-AI/
├── main.py                          #Main system orchestrator
├── config.py                        #Configuration settings
├── neo4j.py                         #Initial graph setup
├── spo_training_manager.py          #SPO relationship prediction
├── requirement.txt                  #Required packages
└── README.md                        #Project documentation
```


## Key Differences from Traditional RAG

RelGraph AI differs from traditional RAG systems by:

1. **Relationship Focus**: Specifically designed for predicting relationships between entities
2. **Session Accuracy**: Tracks prediction accuracy across conversations
3. **Hybrid Prediction**: Combines real database lookups with learned patterns
4. **Interactive Testing**: Provides manual validation interface for relationship predictions
5. **Graph-Native**: Built specifically for Neo4j knowledge graphs rather than document retrieval


## Contributing

This project welcomes contributions in the following areas:
- Additional entity types and relationship patterns
- Enhanced prediction algorithms and accuracy improvements
- Performance optimizations for large-scale graphs
- Integration with additional LLM providers
- Advanced accuracy measurement techniques


## License

This project is available for educational and research purposes. Please ensure compliance with Neo4j, Ollama, and other component licenses when deploying in production environments.


## Support

For technical support:
1. Ensure all dependencies are properly installed
2. Verify Neo4j connection and data availability
3. Check configuration settings match your environment
4. Test with smaller datasets before scaling up
5. Use the interactive testing feature to validate relationship predictions


RelGraph AI provides specialized relationship prediction capabilities with real-time accuracy tracking, making it ideal for applications requiring precise understanding of entity relationships in knowledge graphs.
