import csv
import pandas as pd
from neo4j import GraphDatabase
import logging
import os

#Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveNeo4jManager:
    def __init__(self, uri, user, password, database):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
    
    def close(self):
        self.driver.close()
    
    def clear_database(self):
        """Clear all node and relationship from the database"""
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared")
    
    def create_entity_relationship(self, tx, node1_label, node1_name, relationship_type, node2_label, node2_name):
        """Create a relationship between entity node"""
        query = f"""
        MERGE (a:{node1_label} {{name: $node1_name}})
        MERGE (b:{node2_label} {{name: $node2_name}})
        MERGE (a)-[r:{relationship_type}]->(b)
        RETURN a, r, b
        """
        result = tx.run(query, node1_name=node1_name, node2_name=node2_name)
        return result.single()
    
    def create_dataset_node(self, tx, label, properties):
        """Create a dataset node with given label and property"""
        properties_str = ", ".join([f"{key}: ${key}" for key in properties.keys()])
        query = f"CREATE (n:{label} {{{properties_str}}}) RETURN n"
        result = tx.run(query, **properties)
        return result.single()
    
    def detect_encoding(self, file_path):
        """Detect the encoding of CSV file"""
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']
        
        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read()
                logger.info(f"Successfully detected encoding for {file_path}: {encoding}")
                return encoding
            except UnicodeDecodeError:
                continue
        
        logger.warning(f"Could not detect encoding for {file_path}, defaulting to 'latin-1'")
        return 'latin-1'
    
    def import_single_csv(self, csv_file_path, dataset_label):
        """Import a single CSV file as node with specified label"""
        
        if not os.path.exists(csv_file_path):
            logger.error(f"CSV file not found: {csv_file_path}")
            return False, 0
        
        try:
            #Detect and use appropriate encoding
            encoding = self.detect_encoding(csv_file_path)
            
            try:
                df = pd.read_csv(csv_file_path, encoding=encoding)
            except UnicodeDecodeError:
                logger.warning(f"UTF-8 failed for {csv_file_path}, trying latin-1 encoding...")
                df = pd.read_csv(csv_file_path, encoding='latin-1')
            except Exception:
                logger.warning(f"Standard encoding failed for {csv_file_path}, trying with error handling...")
                df = pd.read_csv(csv_file_path, encoding='utf-8', errors='ignore')
            
            #Clean column name
            df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('[^A-Za-z0-9_]', '', regex=True)
            
            logger.info(f"{dataset_label}: Found CSV with {len(df)} rows and {len(df.columns)} columns")
            logger.info(f"{dataset_label} Columns: {list(df.columns)}")
            
            #Import data to Neo4j
            with self.driver.session(database=self.database) as session:
                node_count = 0
                
                for index, row in df.iterrows():
                    properties = {}
                    for col in df.columns:
                        value = row[col]
                        if pd.isna(value):
                            properties[col] = None
                        elif isinstance(value, (int, float)):
                            properties[col] = value
                        else:
                            clean_value = str(value).encode('utf-8', errors='ignore').decode('utf-8')
                            properties[col] = clean_value
                    
                    #Add identifier
                    properties['row_id'] = index + 1
                    properties['dataset_source'] = dataset_label
                    properties['unique_id'] = f"{dataset_label}_row_{index + 1}"
                    
                    try:
                        session.execute_write(self.create_dataset_node, dataset_label, properties)
                        node_count += 1
                    except Exception as e:
                        logger.error(f"Error creating {dataset_label} node for row {index + 1}: {e}")
                
                logger.info(f"Successfully imported {node_count} {dataset_label} nodes to Neo4j")
                return True, node_count
                
        except Exception as e:
            logger.error(f"Error reading {dataset_label} CSV file: {e}")
            return False, 0
    
    def import_all_datasets(self, dataset_paths):
        """Import all four dataset"""
        results = {}
        
        for dataset_name, file_path in dataset_paths.items():
            logger.info(f"Importing {dataset_name}...")
            success, count = self.import_single_csv(file_path, dataset_name)
            results[dataset_name] = {'success': success, 'count': count}
        
        return results
    
    def create_dataset_to_platform_connections(self):
        """Create connection between dataset and their corresponding platform"""
        
        with self.driver.session(database=self.database) as session:
            
            logger.info("Creating dataset-to-platform connection...")
            
            try:
                #InsightUX represent Dataset1
                session.run("""
                    MERGE (platform:Platform {name: "InsightUX"})
                    WITH platform
                    MATCH (d:Dataset1)
                    MERGE (d)-[:SOURCED_FROM]->(platform)
                """)
                
                #Exsight represent Dataset2
                session.run("""
                    MERGE (platform:Platform {name: "Exsight"})
                    WITH platform
                    MATCH (d:Dataset2)
                    MERGE (d)-[:SOURCED_FROM]->(platform)
                """)
                
                #Afkari represent Dataset3
                session.run("""
                    MERGE (platform:Platform {name: "Afkari"})
                    WITH platform
                    MATCH (d:Dataset3)
                    MERGE (d)-[:SOURCED_FROM]->(platform)
                """)
                
                #InnovateX represent Dataset4
                session.run("""
                    MERGE (platform:Platform {name: "InnovateX"})
                    WITH platform
                    MATCH (d:Dataset4)
                    MERGE (d)-[:SOURCED_FROM]->(platform)
                """)
                
                logger.info("Dataset-to-platform connection created successfully!")
                
            except Exception as e:
                logger.error(f"Error creating dataset-platform connection: {e}")
    
    def create_comprehensive_dataset_connections(self):
        """Create relationship between all four dataset"""
        
        with self.driver.session(database=self.database) as session:
            
            logger.info("Creating inter-dataset connection for all 4 dataset...")
            
            try:
                #Corresponding row connection
                result1 = session.run("""
                    MATCH (d1:Dataset1), (d2:Dataset2)
                    WHERE d1.row_id = d2.row_id
                    MERGE (d1)-[:CORRESPONDS_TO]->(d2)
                    RETURN count(*) as connections
                """)
                count1 = result1.single()["connections"]
                logger.info(f"Created {count1} Dataset1->Dataset2 CORRESPONDS_TO relationships")
                
                result2 = session.run("""
                    MATCH (d2:Dataset2), (d3:Dataset3)
                    WHERE d2.row_id = d3.row_id
                    MERGE (d2)-[:CORRESPONDS_TO]->(d3)
                    RETURN count(*) as connections
                """)
                count2 = result2.single()["connections"]
                logger.info(f"Created {count2} Dataset2->Dataset3 CORRESPONDS_TO relationships")
                
                result3 = session.run("""
                    MATCH (d3:Dataset3), (d4:Dataset4)
                    WHERE d3.row_id = d4.row_id
                    MERGE (d3)-[:CORRESPONDS_TO]->(d4)
                    RETURN count(*) as connections
                """)
                count3 = result3.single()["connections"]
                logger.info(f"Created {count3} Dataset3->Dataset4 CORRESPONDS_TO relationships")
                
                #Cross-dataset connection
                session.run("""
                    MATCH (d1:Dataset1), (d3:Dataset3)
                    WHERE d1.row_id = d3.row_id
                    MERGE (d1)-[:SPANS_TO]->(d3)
                """)
                
                session.run("""
                    MATCH (d1:Dataset1), (d4:Dataset4)
                    WHERE d1.row_id = d4.row_id
                    MERGE (d1)-[:FULL_SPAN]->(d4)
                """)
                
                session.run("""
                    MATCH (d2:Dataset2), (d4:Dataset4)
                    WHERE d2.row_id = d4.row_id
                    MERGE (d2)-[:CROSS_SPAN]->(d4)
                """)
                
                #Sequential flow connection
                session.run("""
                    MATCH (d1:Dataset1), (d2:Dataset2)
                    WHERE d2.row_id = d1.row_id + 1
                    MERGE (d1)-[:FLOWS_TO]->(d2)
                """)
                
                session.run("""
                    MATCH (d2:Dataset2), (d3:Dataset3)
                    WHERE d3.row_id = d2.row_id + 1
                    MERGE (d2)-[:FEEDS_INTO]->(d3)
                """)
                
                session.run("""
                    MATCH (d3:Dataset3), (d4:Dataset4)
                    WHERE d4.row_id = d3.row_id + 1
                    MERGE (d3)-[:TRANSITIONS_TO]->(d4)
                """)
                
                logger.info("All inter-dataset connections created successfully!")
                
            except Exception as e:
                logger.error(f"Error creating dataset connections: {e}")
    
    def create_entity_relationship_model(self):
        """Create the comprehensive entity-relationship model with updated platforms"""
        
        with self.driver.session(database=self.database) as session:
            
            logger.info("Creating enhanced entity-relationship model...")
            
            #Core entity relationship
            relationships = [
                #User interaction-   (User)-[:RAISED]->(Issue)
                ("User", "John_Smith", "RAISED", "Issue", "Performance_Issue"),
                ("User", "Jane_Doe", "RAISED", "Issue", "UI_Bug_Issue"),
                ("User", "Mike_Johnson", "RAISED", "Issue", "Login_Issue"),
                ("User", "Sarah_Wilson", "RAISED", "Issue", "Data_Sync_Issue"),
                ("User", "Alice_Manager", "RAISED", "Issue", "Dashboard_Loading_Issue"),
                
                #Analyst work-   (Analyst)-[:AUTHORED]->(Trend)
                ("Analyst", "Sam_Analyst", "AUTHORED", "Trend", "Remote_Diagnostics_Trend"),
                ("Analyst", "Lisa_Researcher", "AUTHORED", "Trend", "AI_Enabled_Workflows_Trend"),
                ("Analyst", "Arjun_Data", "AUTHORED", "Trend", "Predictive_Maintenance_Trend"),
                ("Analyst", "Meena_Insights", "AUTHORED", "Trend", "Self_Serve_Dashboards_Trend"),
                ("Analyst", "Dev_Analytics", "AUTHORED", "Trend", "Edge_AI_Trend"),
                
                #Contributor proposal-   (Contributor)-[:PROPOSED]->(Idea)
                ("Contributor", "Priya_Innovation", "PROPOSED", "Idea", "Smart_Filter_Optimization"),
                ("Contributor", "Aman_Product", "PROPOSED", "Idea", "Auto_Triage_Complaints"),
                ("Contributor", "Leena_AI", "PROPOSED", "Idea", "Explainable_AI_Decisions"),
                ("Contributor", "Rina_Marketing", "PROPOSED", "Idea", "AI_Customer_Segmentation"),
                ("Contributor", "Arvind_Enterprise", "PROPOSED", "Idea", "Cloud_Collaboration_Tools"),
                
                #Impact relationship-   (Idea)-[:HAS_IMPACT_ON]->(ImpactArea)
                ("Idea", "Smart_Filter_Optimization", "HAS_IMPACT_ON", "ImpactArea", "Efficiency"),
                ("Idea", "Auto_Triage_Complaints", "HAS_IMPACT_ON", "ImpactArea", "Compliance"),
                ("Idea", "Explainable_AI_Decisions", "HAS_IMPACT_ON", "ImpactArea", "Governance"),
                ("Idea", "AI_Customer_Segmentation", "HAS_IMPACT_ON", "ImpactArea", "Personalization"),
                ("Idea", "Cloud_Collaboration_Tools", "HAS_IMPACT_ON", "ImpactArea", "Productivity"),
                
                #Attachment relationship-   (Issue|Idea|Trend)-[:HAS_ATTACHMENT]->(Attachment)
                ("Issue", "Performance_Issue", "HAS_ATTACHMENT", "Attachment", "login_issue_mp4"),
                ("Issue", "UI_Bug_Issue", "HAS_ATTACHMENT", "Attachment", "dashboard_request_pdf"),
                ("Idea", "Smart_Filter_Optimization", "HAS_ATTACHMENT", "Attachment", "filter_idea_pitch_mp4"),
                ("Idea", "AI_Customer_Segmentation", "HAS_ATTACHMENT", "Attachment", "ai_segmentation_pptx"),
                ("Trend", "Remote_Diagnostics_Trend", "HAS_ATTACHMENT", "Attachment", "remote_diag_trend_pdf"),
                ("Trend", "AI_Enabled_Workflows_Trend", "HAS_ATTACHMENT", "Attachment", "ai_workflow_chart_mp4"),
                
                #Media type relationship-   (Attachment)-[:IS_OF_TYPE]->(MediaType)
                ("Attachment", "login_issue_mp4", "IS_OF_TYPE", "MediaType", "Video"),
                ("Attachment", "dashboard_request_pdf", "IS_OF_TYPE", "MediaType", "Document"),
                ("Attachment", "filter_idea_pitch_mp4", "IS_OF_TYPE", "MediaType", "Video"),
                ("Attachment", "ai_segmentation_pptx", "IS_OF_TYPE", "MediaType", "Document"),
                ("Attachment", "remote_diag_trend_pdf", "IS_OF_TYPE", "MediaType", "Document"),
                ("Attachment", "ai_workflow_chart_mp4", "IS_OF_TYPE", "MediaType", "Video"),
                
                #Regional observation-   (Trend)-[:OBSERVED_IN]->(Region)
                ("Trend", "Remote_Diagnostics_Trend", "OBSERVED_IN", "Region", "APAC"),
                ("Trend", "AI_Enabled_Workflows_Trend", "OBSERVED_IN", "Region", "Global"),
                ("Trend", "Predictive_Maintenance_Trend", "OBSERVED_IN", "Region", "India"),
                ("Trend", "Self_Serve_Dashboards_Trend", "OBSERVED_IN", "Region", "US"),
                ("Trend", "Edge_AI_Trend", "OBSERVED_IN", "Region", "Europe"),
                
                #Domain relationship-   (Issue|Trend|Idea)-[:BELONGS_TO]->(Domain)
                ("Issue", "Performance_Issue", "BELONGS_TO", "Domain", "Technical"),
                ("Issue", "UI_Bug_Issue", "BELONGS_TO", "Domain", "User_Interface"),
                ("Issue", "Login_Issue", "BELONGS_TO", "Domain", "Security"),
                ("Issue", "Data_Sync_Issue", "BELONGS_TO", "Domain", "Data_Management"),
                ("Issue", "Dashboard_Loading_Issue", "BELONGS_TO", "Domain", "Performance"),
                ("Trend", "Remote_Diagnostics_Trend", "BELONGS_TO", "Domain", "Healthcare"),
                ("Trend", "AI_Enabled_Workflows_Trend", "BELONGS_TO", "Domain", "Manufacturing"),
                ("Trend", "Predictive_Maintenance_Trend", "BELONGS_TO", "Domain", "Utilities"),
                ("Trend", "Self_Serve_Dashboards_Trend", "BELONGS_TO", "Domain", "Retail"),
                ("Trend", "Edge_AI_Trend", "BELONGS_TO", "Domain", "Transportation"),
                ("Idea", "Smart_Filter_Optimization", "BELONGS_TO", "Domain", "Water"),
                ("Idea", "Auto_Triage_Complaints", "BELONGS_TO", "Domain", "MedTech"),
                ("Idea", "Explainable_AI_Decisions", "BELONGS_TO", "Domain", "Insurance"),
                ("Idea", "AI_Customer_Segmentation", "BELONGS_TO", "Domain", "Marketing"),
                ("Idea", "Cloud_Collaboration_Tools", "BELONGS_TO", "Domain", "Enterprise"),
                
                #Platform origin with updated platform-   (Idea|Trend|Issue)-[:ORIGINATED_FROM]->(Platform)
                ("Issue", "Performance_Issue", "ORIGINATED_FROM", "Platform", "InsightUX"),
                ("Issue", "UI_Bug_Issue", "ORIGINATED_FROM", "Platform", "Exsight"),
                ("Issue", "Login_Issue", "ORIGINATED_FROM", "Platform", "Afkari"),
                ("Issue", "Data_Sync_Issue", "ORIGINATED_FROM", "Platform", "InnovateX"),
                ("Issue", "Dashboard_Loading_Issue", "ORIGINATED_FROM", "Platform", "InsightUX"),
                ("Trend", "Remote_Diagnostics_Trend", "ORIGINATED_FROM", "Platform", "InsightUX"),
                ("Trend", "AI_Enabled_Workflows_Trend", "ORIGINATED_FROM", "Platform", "InsightUX"),
                ("Trend", "Predictive_Maintenance_Trend", "ORIGINATED_FROM", "Platform", "InsightUX"),
                ("Trend", "Self_Serve_Dashboards_Trend", "ORIGINATED_FROM", "Platform", "InsightUX"),
                ("Trend", "Edge_AI_Trend", "ORIGINATED_FROM", "Platform", "InsightUX"),
                ("Idea", "Smart_Filter_Optimization", "ORIGINATED_FROM", "Platform", "Afkari"),
                ("Idea", "Auto_Triage_Complaints", "ORIGINATED_FROM", "Platform", "Afkari"),
                ("Idea", "Explainable_AI_Decisions", "ORIGINATED_FROM", "Platform", "Afkari"),
                ("Idea", "AI_Customer_Segmentation", "ORIGINATED_FROM", "Platform", "InnovateX"),
                ("Idea", "Cloud_Collaboration_Tools", "ORIGINATED_FROM", "Platform", "InnovateX"),
                
                #Status relationship-   (Issue|Idea|Trend)-[:HAS_STATUS]->(Status)
                ("Issue", "Performance_Issue", "HAS_STATUS", "Status", "Open"),
                ("Issue", "UI_Bug_Issue", "HAS_STATUS", "Status", "Closed"),
                ("Issue", "Login_Issue", "HAS_STATUS", "Status", "Open"),
                ("Issue", "Data_Sync_Issue", "HAS_STATUS", "Status", "Open"),
                ("Issue", "Dashboard_Loading_Issue", "HAS_STATUS", "Status", "Open"),
                ("Idea", "Smart_Filter_Optimization", "HAS_STATUS", "Status", "Under_Review"),
                ("Idea", "Auto_Triage_Complaints", "HAS_STATUS", "Status", "Approved"),
                ("Idea", "Explainable_AI_Decisions", "HAS_STATUS", "Status", "Draft"),
                ("Idea", "AI_Customer_Segmentation", "HAS_STATUS", "Status", "In_Pipeline"),
                ("Idea", "Cloud_Collaboration_Tools", "HAS_STATUS", "Status", "Approved"),
                ("Trend", "Remote_Diagnostics_Trend", "HAS_STATUS", "Status", "Published"),
                ("Trend", "AI_Enabled_Workflows_Trend", "HAS_STATUS", "Status", "Published"),
                ("Trend", "Predictive_Maintenance_Trend", "HAS_STATUS", "Status", "Published"),
                ("Trend", "Self_Serve_Dashboards_Trend", "HAS_STATUS", "Status", "Published"),
                ("Trend", "Edge_AI_Trend", "HAS_STATUS", "Status", "Published"),
            ]
            
            #Create relationship
            for source_label, source_name, rel_type, target_label, target_name in relationships:
                try:
                    session.execute_write(
                        self.create_entity_relationship,
                        source_label, source_name,
                        rel_type,
                        target_label, target_name
                    )
                    logger.info(f"Created: ({source_name})-[{rel_type}]->({target_name})")
                except Exception as e:
                    logger.error(f"Error creating entity relationship: {e}")
            
            logger.info("Enhanced entity-relationship model created successfully!")
    
    def verify_comprehensive_model(self, dataset_results):
        """Verification of all node and relationship"""
        with self.driver.session(database=self.database) as session:
            
            print("\n" + "="*80)
            print("KNOWLEDGE GRAPH VERIFICATION - 4 DATASET WITH INNOVATEX")
            print("="*80)
            
            #Dataset node count
            print(f"\nDATASET NODE COUNT:")
            total_dataset_nodes = 0
            for dataset_name in ["Dataset1", "Dataset2", "Dataset3", "Dataset4"]:
                count = session.run(f"MATCH (n:{dataset_name}) RETURN count(n) as count").single()["count"]
                print(f"- {dataset_name}: {count} nodes")
                total_dataset_nodes += count
            
            print(f"Total Dataset Node: {total_dataset_nodes}")
            
            #Platform verification with dataset connection
            print(f"\nPLATFORM-DATASET CONNECTION:")
            platform_connections = session.run("""
                MATCH (platform:Platform)<-[:SOURCED_FROM]-(dataset)
                RETURN platform.name as platform_name, labels(dataset)[0] as dataset_label, count(dataset) as count
                ORDER BY platform_name
            """)
            
            for record in platform_connections:
                print(f"- {record['platform_name']} <- {record['dataset_label']}: {record['count']} nodes")
            
            #Entity model node count
            print(f"\nENTITY MODEL NODE COUNT:")
            entity_labels = ["User", "Analyst", "Contributor", "Feature", "Issue", "Trend", 
                           "Insight", "Idea", "Domain", "Region", "ImpactArea", "Status", 
                           "Attachment", "MediaType", "DataSource", "Platform"]
            
            total_entity_nodes = 0
            for label in entity_labels:
                count = session.run(f"MATCH (n:{label}) RETURN count(n) as count").single()["count"]
                if count > 0:
                    print(f"- {label}: {count} nodes")
                    total_entity_nodes += count
            
            print(f"Total Entity Model Node: {total_entity_nodes}")
            
            #Platform distribution in entity model
            print(f"\nPLATFORM USAGE IN ENTITY MODEL:")
            platform_usage = session.run("""
                MATCH (entity)-[:ORIGINATED_FROM]->(platform:Platform)
                RETURN platform.name as platform_name, labels(entity)[0] as entity_type, count(entity) as count
                ORDER BY platform_name, entity_type
            """)
            
            for record in platform_usage:
                print(f"- {record['platform_name']}: {record['count']} {record['entity_type']} entities")
            
            #Relationship type verification
            print(f"\nENTITY RELATIONSHIP VERIFICATION:")
            specific_patterns = [
                ("User", "RAISED", "Issue"),
                ("Analyst", "AUTHORED", "Trend"),
                ("Contributor", "PROPOSED", "Idea"),
                ("Idea", "HAS_IMPACT_ON", "ImpactArea"),
                ("Issue", "HAS_ATTACHMENT", "Attachment"),
                ("Attachment", "IS_OF_TYPE", "MediaType"),
                ("Trend", "OBSERVED_IN", "Region"),
                ("Issue", "BELONGS_TO", "Domain"),
                ("Idea", "ORIGINATED_FROM", "Platform"),
                ("Issue", "HAS_STATUS", "Status"),
            ]
            
            for source_label, rel_type, target_label in specific_patterns:
                count = session.run(f"""
                    MATCH (a:{source_label})-[r:{rel_type}]->(b:{target_label})
                    RETURN count(r) as count
                """).single()["count"]
                print(f"- ({source_label})-[{rel_type}]->({target_label}): {count} relationships")
            
            #Total statistic
            total_nodes = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            total_relationships = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            
            print(f"\nOVERALL STATISTIC:")
            print(f"- Total Node: {total_nodes}")
            print(f"- Total Relationship: {total_relationships}")
            print(f"- Dataset Node: {total_dataset_nodes}")
            print(f"- Entity Model Node: {total_entity_nodes}")
            
            print(f"\nNEO4J BROWSER QUERY:")
            print("View all platform: MATCH (p:Platform) RETURN p")
            print("View platform connection: MATCH (p:Platform)<-[:SOURCED_FROM]-(d) RETURN p,d")
            print("View entity model: MATCH (n)-[r]->(m) WHERE NOT labels(n)[0] IN ['Dataset1','Dataset2','Dataset3','Dataset4'] RETURN n,r,m")
            print("View InnovateX entitie: MATCH (n)-[:ORIGINATED_FROM]->(p:Platform {name: 'InnovateX'}) RETURN n,p")

def main():
    #Connection parameter
    uri = "neo4j://127.0.0.1:7687"
    user = "neo4j"
    password = "Harsh@123"
    database = "testdb"
    
    #Dataset file path
    dataset_paths = {
        "Dataset1": r"C:\Users\Harsh\Desktop\neo\DataConnect\Dataset1.csv",
        "Dataset2": r"C:\Users\Harsh\Desktop\neo\DataConnect\Dataset2.csv", 
        "Dataset3": r"C:\Users\Harsh\Desktop\neo\DataConnect\Dataset3.csv",
        "Dataset4": r"C:\Users\Harsh\Desktop\neo\DataConnect\Dataset4.csv"
    }
    
    #Create comprehensive manager
    manager = ComprehensiveNeo4jManager(uri, user, password, database)
    
    try:
        logger.info("Starting Neo4j knowledge graph creation with InnovateX platform...")
        
        #Clear database for fresh start
        manager.clear_database()
        
        #Step 1: Import all four dataset
        logger.info("Step 1: Importing all four dataset...")
        dataset_results = manager.import_all_datasets(dataset_paths)
        
        #Step 2: Create dataset connection
        logger.info("Step 2: Creating inter-dataset connection...")
        manager.create_comprehensive_dataset_connections()
        
        #Step 3: Create dataset-to-platform connection
        logger.info("Step 3: Creating dataset-to-platform connection...")
        manager.create_dataset_to_platform_connections()
        
        #Step 4: Create the enhanced entity-relationship model
        logger.info("Step 4: Creating enhanced entity-relationship model...")
        manager.create_entity_relationship_model()
        
        #Step 5: Comprehensive verification
        logger.info("Step 5: Verifying the complete knowledge graph...")
        manager.verify_comprehensive_model(dataset_results)
        
        print(f"\nCOMPLETE KNOWLEDGE GRAPH WITH INNOVATEX PLATFORM CREATED SUCCESSFULLY!")
        print("Your testdb database now contain:")
        print("- All 4 dataset with platform connection:")
        print("  * InsightUX (Dataset1)")
        print("  * Exsight (Dataset2)")
        print("  * Afkari (Dataset3)")
        print("  * InnovateX (Dataset4)")
        print("- Complete entity-relationship model with all specified relationship")
        print("- Enhanced platform integration showing dataset origin")
        print("- Comprehensive inter-dataset connection")
        
    except Exception as e:
        logger.error(f"Error during knowledge graph creation: {e}")
    finally:
        manager.close()

if __name__ == "__main__":
    main()