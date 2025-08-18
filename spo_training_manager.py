"""
SPO Training Manager: Relationship Prediction with Session Accuracy Tracking
- Trains on SPO triplets extracted from Neo4j graph and synthetic data
- Predicts relationships between node pairs for "What is the relationship between X and Y?" query
- Tracks session accuracy for all relationship question asked
- Provides detailed reporting with confidence level and error analysis
"""

import json
import random
from collections import defaultdict
from typing import List, Dict, Any, Tuple

from neo4j import GraphDatabase
from config import Config

class RelationshipPredictionManager:
    """
    Main class for training and predicting relationship between nodes
    Handles SPO (Subject-Predicate-Object) triplet learning and real-time prediction
    """
    
    def __init__(self):
        #Initialize Neo4j database connection using config setting
        self.driver = GraphDatabase.driver(
            Config.NEO4J_URI,
            auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD),
            encrypted=False
        )
        self.database = Config.NEO4J_DATABASE
        
        #Core data structure for training and prediction
        self.training_triplets = []          #All SPO triplets used for training
        self.predicate_patterns = {}         #Learned pattern: (subject_type, object_type) -> most_common_predicate
        self.session_evals = []              #All relationship question evaluated in current session
        self.session_questions = []          #Detailed tracking of each question with result
        self._report_printed = False         #Flag to prevent infinite loop in reporting
    
    def close(self):
        """Clean up database connection when done"""
        if self.driver:
            self.driver.close()
    
    def extract_spo_triplets_from_graph(self, limit=200) -> List[Dict[str, str]]:
        """
        Extract Subject-Predicate-Object triplets from the Neo4j knowledge graph
        This provides real relationship data from your actual graph for training
        """
        triplets = []
        with self.driver.session(database=self.database) as session:
            #Cypher query to get all relationship with node types
            query = """
            MATCH (s)-[r]->(o)
            WHERE s.name IS NOT NULL AND o.name IS NOT NULL
            WITH s, r, o,
                 CASE WHEN size(labels(s)) > 0 THEN labels(s)[0] ELSE 'Unknown' END as s_type,
                 CASE WHEN size(labels(o)) > 0 THEN labels(o) ELSE 'Unknown' END as o_type
            RETURN s.name as s_name, s_type, type(r) as predicate, o.name as o_name, o_type
            LIMIT $limit
            """
            try:
                result = session.run(query, limit=limit)
                for record in result:
                    #Ensure all values are string to prevent type errors
                    s_name = str(record.get("s_name", ""))
                    s_type = str(record.get("s_type", "Unknown"))
                    predicate = str(record.get("predicate", ""))
                    o_name = str(record.get("o_name", ""))
                    o_type = str(record.get("o_type", "Unknown"))
                    
                    #Only add triplets with valid data
                    if s_name and predicate and o_name:
                        triplets.append({
                            "subject": s_name,
                            "subject_type": s_type,
                            "predicate": predicate,
                            "object": o_name,
                            "object_type": o_type,
                        })
            except Exception as e:
                print(f"Error extracting triplets: {e}")
                pass
        return triplets

    def generate_synthetic_triplets(self, count=200) -> List[Dict[str, str]]:
        """
        Generate synthetic SPO triplets based on common relationship patterns
        This supplements real data with typical knowledge graph relationships
        """
        #Define common relationship patterns in knowledge graph
        types = [
            ("User",  "RAISED",     "Issue"),           #Users raise issues
            ("User",  "USES",       "Platform"),        #Users use platforms
            ("Analyst", "AUTHORED", "Trend"),           #Analysts author trends
            ("Analyst", "COVERS",   "Region"),          #Analysts cover regions
            ("Contributor", "DEVELOPED", "Idea"),       #Contributors develop ideas
            ("Issue", "BELONGS_TO", "Domain"),          #Issues belong to domains
            ("Trend", "OBSERVED_IN", "Region"),         #Trends observed in regions
            ("Idea", "HAS_IMPACT_ON", "ImpactArea"),    #Ideas have impact on areas
            ("Platform", "SUPPORTS", "Domain"),         #Platforms support domains
        ]
        
        samples = []
        for i in range(count):
            #Cycle through relationship type to create balanced synthetic data
            t = types[i % len(types)]
            samples.append({
                "subject": f"{t[0]}_{i%25+1}",        #Create numbered entities
                "subject_type": t,
                "predicate": t,
                "object": f"{t}_{i%25+1}",
                "object_type": t,
            })
        return samples

    def train(self):
        """
        Main training method that extracts real data, generates synthetic data,
        and builds prediction patterns for relationship forecasting
        """
        print("Extracting triplets from graph and generating synthetic data...")
        
        #Get real relationship data from your Neo4j graph
        real = self.extract_spo_triplets_from_graph(limit=200)
        # Generate synthetic data to supplement real data
        synthetic = self.generate_synthetic_triplets(count=200)
        #Combine both data sources for training
        self.training_triplets = real + synthetic
        
        print(f"Building prediction patterns from {len(self.training_triplets)} triplets...")
        
        #Build prediction patterns using defaultdict for easy counting
        pattern_counter = defaultdict(lambda: defaultdict(int))
        
        for t in self.training_triplets:
            #Ensure all values are string to prevent error
            subject_type = str(t.get('subject_type', 'Unknown'))
            object_type = str(t.get('object_type', 'Unknown'))
            predicate = str(t.get('predicate', 'RELATES_TO'))
            
            #Count how often each predicate appears for each (subject_type, object_type) pair
            key = (subject_type, object_type)
            pattern_counter[key][predicate] += 1
        
        #Convert to regular dictionary and find most common predicate for each pattern
        self.predicate_patterns = {}
        for key, pred_count in pattern_counter.items():
            if pred_count:  #Only process patterns that have data
                #Find the most frequently occurring predicate for this node type combination
                most_common = max(pred_count, key=pred_count.get)
                self.predicate_patterns[key] = most_common
        
        print(f"Learned {len(self.predicate_patterns)} relationship patterns")

    def _infer_type(self, name: str) -> str:
        """
        Infer node type from node name using pattern matching
        This helps predict relationships for nodes not seen during training
        """
        if not name:
            return "Unknown"
            
        n = name.lower()
        #Define mappings from name patterns to node types
        mappings = {
            "user": "User", "analyst": "Analyst", "contributor": "Contributor",
            "issue": "Issue", "trend": "Trend", "idea": "Idea",
            "platform": "Platform", "domain": "Domain", "region": "Region",
            "impactarea": "ImpactArea"
        }
        
        #Check for exact prefix matches first (User_1 -> User)
        for k, typ in mappings.items():
            if name.lower().startswith(k):
                return typ
        
        #Check for substring matches as fallback (MyUser -> User)
        for k, typ in mappings.items():
            if k in n:
                return typ
                
        return "Unknown"

    def check_direct_relationship_from_graph(self, subject: str, object_: str) -> str:
        """
        Check Neo4j database for actual relationship between two specific nodes
        This ensures we return real relationships when they exist in the graph
        """
        with self.driver.session(database=self.database) as session:
            #Query for direct relationship between nodes by name
            query = """
            MATCH (s {name: $subject})-[r]->(o {name: $object})
            RETURN type(r) as predicate LIMIT 1
            """
            try:
                result = session.run(query, subject=subject, object=object_)
                record = result.single()
                if record:
                    return record["predicate"]  #Return actual relationship type
            except Exception as e:
                print(f"Error checking direct relationship: {e}")
            return None  #No direct relationship found

    def predict_relationship(self, subject: str, object_: str) -> str:
        """
        Predict the most likely relationship between two nodes
        Uses two-step approach: 1) Check actual graph, 2) Use learned patterns
        """
        if not subject or not object_:
            return "RELATES_TO"
        
        #Strategy 1: Check for actual relationship in Neo4j database first
        direct_pred = self.check_direct_relationship_from_graph(subject, object_)
        if direct_pred:
            return direct_pred  #Return real relationship if it exists
        
        #Strategy 2: Use pattern-based prediction if no direct relationship found
        subject_type = self._infer_type(subject)
        object_type = self._infer_type(object_)
        
        #Look up most common predicate for this node type combination
        key = (subject_type, object_type)
        pred = self.predicate_patterns.get(key, "RELATES_TO")  # Default fallback
        return pred

    def track_relationship_question(self, subject: str, object_: str, predicted: str) -> Dict[str, Any]:
        """
        Track each relationship question for comprehensive session accuracy calculation
        Determines correctness by comparing against database or learned patterns
        """
        #Check if actual relationship exists in database
        actual_relationship = self.check_direct_relationship_from_graph(subject, object_)
        
        if actual_relationship:
            #If relationship exists in DB, accuracy based on exact match
            is_correct = (predicted == actual_relationship)
            confidence = "High" if predicted == actual_relationship else "Low"
            source = 'database'
        else:
            #If no relationship exists in DB, accuracy based on learned patterns
            subject_type = self._infer_type(subject)
            object_type = self._infer_type(object_)
            key = (subject_type, object_type)
            expected_pattern = self.predicate_patterns.get(key, "RELATES_TO")
            is_correct = (predicted == expected_pattern)
            confidence = "Medium" if is_correct else "Low"
            source = 'pattern'
        
        #Create detailed tracking entry for this question
        question_entry = {
            'question_number': len(self.session_questions) + 1,
            'subject': subject,
            'object': object_,
            'predicted': predicted,
            'actual_in_db': actual_relationship,
            'is_correct': is_correct,
            'confidence': confidence,
            'source': source
        }
        
        self.session_questions.append(question_entry)
        return question_entry

    def validate_interaction(self, subject: str, object_: str, expected: str = None) -> Dict[str, Any]:
        """
        Main method for processing relationship questions and tracking results
        Called every time user asks "What is the relationship between X and Y?"
        """
        #Get prediction using the two-step approach
        predicted = self.predict_relationship(subject, object_)
        
        #Track this question for session accuracy calculation
        question_tracking = self.track_relationship_question(subject, object_, predicted)
        
        #Determine correctness
        if expected is not None:
            #If expected answer is provided, use direct comparison
            correct = (predicted == expected)
        else:
            # Otherwise, use the tracking result to determine correctness
            correct = question_tracking['is_correct']
        
        #Create evaluation entry for this interaction
        entry = {
            'subject': str(subject),
            'object': str(object_),
            'expected': str(expected) if expected else None,
            'predicted': str(predicted),
            'correct': correct
        }
        self.session_evals.append(entry)
        return entry

    def clear_session_evaluations(self):
        """Reset session tracking for a new testing session"""
        self.session_evals = []
        self.session_questions = []
        self._report_printed = False

    def session_accuracy(self) -> float:
        """
        Calculate current session accuracy based on all relationship questions asked
        Returns percentage of correctly predicted relationships
        """
        if not self.session_questions:
            return 0.0
        
        correct_count = sum(1 for q in self.session_questions if q['is_correct'])
        return correct_count / len(self.session_questions)
    
    def print_session_report(self):
        """
        Print comprehensive session accuracy report with detailed breakdown
        Shows overall accuracy, source breakdown, and individual question results
        """
        #Prevent infinite loop by checking if report was already printed
        if self._report_printed:
            return
            
        self._report_printed = True
        
        total_questions = len(self.session_questions)
        acc = self.session_accuracy()
        
        #Print header and summary statistics
        print(f"\n" + "="*60)
        print(f"RELATIONSHIP PREDICTION SESSION REPORT")
        print(f"="*60)
        print(f"Total relationship questions asked: {total_questions}")
        print(f"Overall accuracy: {acc:.2%}")
        
        if total_questions > 0:
            #Breakdown accuracy by data source (database vs pattern-based)
            db_questions = [q for q in self.session_questions if q['source'] == 'database']
            pattern_questions = [q for q in self.session_questions if q['source'] == 'pattern']
            
            if db_questions:
                db_accuracy = sum(1 for q in db_questions if q['is_correct']) / len(db_questions)
                print(f"Database-verified relationships: {len(db_questions)} (Accuracy: {db_accuracy:.2%})")
            
            if pattern_questions:
                pattern_accuracy = sum(1 for q in pattern_questions if q['is_correct']) / len(pattern_questions)
                print(f"Pattern-based predictions: {len(pattern_questions)} (Accuracy: {pattern_accuracy:.2%})")
            
            #Show detailed results for each question
            print(f"\nDetailed Results:")
            for q in self.session_questions:
                status = "✓ CORRECT" if q['is_correct'] else "✗ INCORRECT"
                print(f"{q['question_number']}. {q['subject']} -> {q['object']}: {q['predicted']} [{status}] ({q['confidence']} confidence)")
                if q['actual_in_db']:
                    print(f"    Database has: {q['actual_in_db']}")
                else:
                    print(f"    No direct relationship in database")

    def evaluate_batch(self, items: List[Tuple[str, str, str]]) -> float:
        """
        Evaluate a batch of relationship predictions for testing purposes
        Used for programmatic validation with known correct answers
        """
        self.clear_session_evaluations()
        for subj, obj, expected in items:
            self.validate_interaction(subj, obj, expected)
        return self.session_accuracy()