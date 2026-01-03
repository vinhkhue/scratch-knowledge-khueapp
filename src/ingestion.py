import os
import json
import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neo4j import GraphDatabase
from openai import OpenAI
import config

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants & Prompts ---
CHUNK_SIZE = 3000
MAX_WORKERS = 3  # Parallel LLM calls

SYSTEM_PROMPT_GRAPH = """
You are an expert Data Extractor for a Knowledge Graph about Scratch Programming.
Your task is to extract **Entities** and **Relationships** from the provided text.

RULES:
1. **ENTITIES**:
   - Extract key terms: Concepts (e.g., Loop, Variable), Blocks (e.g., Move 10 steps), UI Elements (e.g., Stage, Sprite).
   - Ignore generic terms (Chapter, Exercise).
   - Output: {"name": "X", "type": "Type", "description": "Short def"}

2. **RELATIONSHIPS**:
   - Identify how these specific entities relate *based on the text*.
   - Use standard types: IS_A, HAS_PART, USES, CONTROLS, DEFINES, RELATED_TO.
   - Output: {"source": "EntityA", "target": "EntityB", "type": "RELATION", "description": "Context"}

3. **OUTPUT FORMAT**:
   Return a single JSON object:
   {
     "entities": [...],
     "relationships": [...]
   }
"""

class GraphIngestor:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI, 
            auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
        )
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.entities = {} # name -> metadata
        self.relationships = []

    def close(self):
        self.driver.close()

    def wipe_database(self):
        """DANGER: Deletes all nodes and relationships."""
        logger.warning("Wiping Neo4j Database...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Database wiped.")

    def _call_llm(self, system_prompt: str, user_content: str) -> Dict:
        """Helper to call OpenAI and parse JSON."""
        try:
            response = self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                response_format={"type": "json_object"},
                temperature=0.0 # Low temp for deterministic extraction
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"LLM Call Failed: {e}")
            return {}

    def extract_from_file(self, file_path: str):
        logger.info(f"Reading file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
        logger.info(f"Split into {len(chunks)} chunks.")

        # Optimization: Process ALL chunks 
        keywords = ["scratch", "lập trình", "khối lệnh", "biến", "vòng lặp", "chương trình"]
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_chunk = {}
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 50: continue

                logger.info(f"Scheduling extraction for chunk {i+1}/{len(chunks)}")
                future_to_chunk[executor.submit(self._call_llm, SYSTEM_PROMPT_GRAPH, chunk)] = chunk

            for future in as_completed(future_to_chunk):
                res = future.result()
                
                # Merge Entities
                if "entities" in res:
                    for ent in res["entities"]:
                        self.entities[ent["name"]] = ent
                
                # Merge Relationships
                if "relationships" in res:
                    self.relationships.extend(res["relationships"])

        logger.info(f"Extracted {len(self.entities)} unique entities.")
        logger.info(f"Extracted {len(self.relationships)} relationships.")

    def ingest_to_neo4j(self):
        logger.info("Ingesting into Neo4j...")
        with self.driver.session() as session:
            # Create Constraints
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
            
            # Ingest Nodes
            logger.info(f"Ingesting {len(self.entities)} nodes...")
            count = 0
            for name, data in self.entities.items():
                session.run(
                    """
                    MERGE (e:Entity {name: $name})
                    SET e.type = $type, 
                        e.description = $description,
                        e.last_updated = datetime()
                    """,
                    name=name, type=data.get("type", "General"), description=data.get("description", "")
                )
                count += 1
            
            # Ingest Relationships
            logger.info(f"Ingesting {len(self.relationships)} relationships...")
            for rel in self.relationships:
                # Ensure simple clean keys
                source = rel.get("source")
                target = rel.get("target") 
                
                if source and target and source in self.entities and target in self.entities:
                    session.run(
                        """
                        MATCH (a:Entity {name: $source})
                        MATCH (b:Entity {name: $target})
                        MERGE (a)-[r:RELATED_TO {type: $type}]->(b)
                        SET r.description = $description
                        """,
                        source=source, target=target, type=rel.get("type", "RELATED_TO"), description=rel.get("description", "")
                    )
                else:
                    pass
                    
        logger.info("Ingestion Complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wipe", action="store_true", help="Wipe DB before ingest")
    parser.add_argument("--file", type=str, help="Ingest only a specific file")
    args = parser.parse_args()

    ingestor = GraphIngestor()
    try:
        if args.wipe:
            ingestor.wipe_database()
        
        # Scan Input Directory
        input_dir = "data/scratch_index/input"
        if not os.path.exists(input_dir):
            os.makedirs(input_dir, exist_ok=True)
            
        if args.file:
            # Process specific file
            fpath = os.path.join(input_dir, args.file)
            if os.path.exists(fpath):
                ingestor.extract_from_file(fpath)
            else:
                logger.error(f"File not found: {fpath}")
        else:
            # Process all files
            files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
            if not files:
                logger.warning(f"No .txt files found in {input_dir}. Please add data.")
            
            for fname in files:
                ingestor.extract_from_file(os.path.join(input_dir, fname))
        
        if ingestor.entities:
            ingestor.ingest_to_neo4j()
        else:
            logger.warning("No entities extracted. Skipping DB ingestion.")
            
    finally:
        ingestor.close()
