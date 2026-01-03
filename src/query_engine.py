import logging
import json
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neo4j import GraphDatabase
from openai import OpenAI
from src.web_search import WebSearch
import config

logger = logging.getLogger(__name__)

class GraphQueryEngine:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI, 
            auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
        )
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.web_search = WebSearch()

    def close(self):
        self.driver.close()

    def _extract_search_intents(self, query: str) -> list[str]:
        """
        Uses LLM to extract specific core concepts/entities being asked about.
        Example: "Cách tạo vòng lặp trong Scratch" -> ["Vòng lặp", "Loop", "Control Blocks"]
        """
        try:
            system_prompt = """
            You are a sub-module for a Search Engine over a Scratch Programming Knowledge Graph.
            Your job is to extract the **Core Concepts** or **Entity Names** from the user's question.
            
            RULES:
            1. Return a JSON object: {"keywords": ["Key1", "Key2"]}.
            2. Extract specific technical terms (e.g., "Loop", "Variable", "Event", "Sprite").
            3. If the user asks about a specific block (e.g., "Move 10 steps"), extract the block name or type.
            4. Include English translations if recognized (e.g. "Vòng lặp" -> "Loop").
            5. **SPLIT** compound queries (e.g. "Types of Blocks" -> ["Block", "Type"]).
            6. Keep it minimal (1-3 keywords).
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini", # Use mini for speed/cost
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            data = json.loads(response.choices[0].message.content)
            keywords = data.get("keywords", [])
            
            # Robustness: If keywords contain spaces, split them too
            final_keywords = []
            for k in keywords:
                if " " in k:
                    final_keywords.extend(k.split())
                else:
                    final_keywords.append(k)
            
            # Unique
            return list(set(final_keywords))

        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}. Falling back to naive matching.")
            # Fallback to naive splitting
            import re
            clean = re.sub(r'[^\w\s]', '', query)
            return [w for w in clean.split() if len(w) > 3]

    def _get_relevant_context(self, query: str) -> tuple[str, dict]:
        """
        Returns: (context_text, graph_data_json)
        graph_data_json structure: {"nodes": [], "edges": []}
        """
        # 1. Extract Intents/Keywords
        search_terms = self._extract_search_intents(query)
        logger.info(f"Search Terms: {search_terms}")
        
        context_lines = []
        graph_nodes = {}  # Dedup by ID/Name
        graph_edges = []
        
        # 2. Weighted Cypher Query
        # Priority: Exact Name Match > Name Contains > Description Contains
        cypher_query = """
        MATCH (e:Entity)
        WITH e, 
             reduce(s=0, term IN $terms | 
                s + CASE 
                    WHEN toLower(e.name) = toLower(term) THEN 10 
                    WHEN toLower(e.name) CONTAINS toLower(term) THEN 5
                    WHEN toLower(e.description) CONTAINS toLower(term) THEN 2
                    ELSE 0 
                END
             ) AS score
        WHERE score > 0
        RETURN e.name, e.description, e.type, score
        ORDER BY score DESC
        LIMIT 5
        """

        with self.driver.session() as session:
            result = session.run(cypher_query, terms=search_terms)
            entities = [(r["e.name"], r["e.description"], r["e.type"]) for r in result]
            
            # For each entity, get 1-hop neighborhood
            for name, desc, label in entities:
                context_lines.append(f"ENTITY: {name} ({desc})")
                graph_nodes[name] = {"id": name, "label": name, "title": desc, "group": label or "Entity"}

                # Get relationships
                rel_cypher = """
                MATCH (a:Entity {name: $name})-[r]->(b:Entity)
                RETURN type(r) as rel_type, b.name as target, b.description as target_desc, b.type as target_type
                LIMIT 5
                """
                rels = session.run(rel_cypher, name=name)
                for r in rels:
                    target_name = r['target']
                    context_lines.append(f"  - {r['rel_type']} -> {target_name} ({r['target_desc']})")
                    
                    # Add target node
                    if target_name not in graph_nodes:
                         graph_nodes[target_name] = {"id": target_name, "label": target_name, "title": r['target_desc'], "group": r['target_type'] or "Entity"}
                    
                    # Add edge
                    graph_edges.append({"source": name, "target": target_name, "label": r['rel_type']})
        
        graph_data = {
            "nodes": list(graph_nodes.values()),
            "edges": graph_edges
        }
        return "\n".join(context_lines), graph_data
        
    def _run_tool_search(self, messages, tools):
        """Helper to handle tool execution loop"""
        response = self.client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=messages,
            tools=tools,
            temperature=0.3
        )
        msg = response.choices[0].message
        
        # Handle Tool Calls
        if msg.tool_calls:
            messages.append(msg) # Add assistant's tool call message
            
            for tool_call in msg.tool_calls:
                if tool_call.function.name == "web_search":
                    args = json.loads(tool_call.function.arguments)
                    q = args.get("query")
                    logger.info(f"LLM invoking Web Search with query: '{q}'")
                    
                    # Execute Search
                    try:
                        search_res = self.web_search.search(q)
                        content = search_res if search_res else "No results found."
                    except Exception as e:
                        content = f"Error performing search: {str(e)}"

                    # Add Tool Output
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": "web_search",
                        "content": content
                    })
            
            # Follow-up completion to answer user
            final_response = self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                temperature=0.3
            )
            return final_response.choices[0].message.content, "Web Search"
            
        return msg.content, "AI Knowledge"

    def search(self, query: str, force_web_search: bool = False) -> tuple[str, dict, str]:
        context, graph_data = self._get_relevant_context(query)
        source = "GraphRAG"
        
        # Tool Definition
        tools = [{
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the internet for up-to-date information about Scratch, programming, or specific blocks.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to send to the search engine (e.g. 'Scratch loop block example')."
                        }
                    },
                    "required": ["query"]
                }
            }
        }]

        # Logic Flow:
        # 1. If Graph Nodes Found -> Use Graph (Preferred).
        # 2. If NO Graph Nodes -> Fallback to Web Search via Tool.
        # 3. If Force Web Search -> Skip Graph, go straight to Tool.

        messages = [
            {"role": "system", "content": """You are a helpful assistant powered by a Knowledge Graph and Web Search.
            PRIORITY:
            1. Use the provided [GRAPH CONTEXT] if available and relevant.
            2. If the context is empty or you need more info, you can call the 'web_search' tool.
            3. Answer in Vietnamese."""}
        ]

        if force_web_search:
             logger.info("Forced Web Search triggered.")
             messages.append({"role": "user", "content": f"Please verify this on the web: {query}"})
             
             text, src = self._run_tool_search(messages, tools)
             return text, {"nodes":[], "edges":[]}, src
        
        # Standard Flow
        if graph_data["nodes"]:
            # Strong Graph Context - Try to answer directly
            messages.append({"role": "user", "content": f"""
            [GRAPH CONTEXT]:
            {context}
            
            USER QUESTION: {query}
            
            Answer based on the context. If it's completely irrelevant, you may use the web_search tool.
            """})
            
            try:
                resp = self.client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=messages,
                    tools=tools, # Allow tool usage if needed
                    temperature=0.3
                )
                msg = resp.choices[0].message
                
                # If LLM decides to use tool immediately
                if msg.tool_calls:
                     logger.info("LLM autonomously chose to use Web Tool despite graph context.")
                     text, src = self._run_tool_search(messages, tools)
                     return text, {"nodes": [], "edges": []}, src

                text = msg.content
                
                # Check for refusal
                refusal_keywords = ["không tìm thấy", "không có thông tin", "xin lỗi"]
                if any(k in text.lower() for k in refusal_keywords):
                     logger.info("GraphRAG refusal detected. Retrying with Web Tool...")
                     # Retry with tool capability
                     messages.append(msg)
                     messages.append({"role": "user", "content": "The graph didn't have it. Please search the web."})
                     text, src = self._run_tool_search(messages, tools)
                     return text, {"nodes": [], "edges": []}, src
                
                return text, graph_data, "GraphRAG"

            except Exception as e:
                logger.error(f"Error in main search: {e}")
                return "Lỗi xử lý.", graph_data, "Error"

        else:
            # Empty Graph -> Auto Fallback to Web Tool
            logger.info("Graph context empty. Auto-switching to Web Tool.")
            messages.append({"role": "user", "content": f"I cannot find information in the database. Please search the web for: {query}"})
            text, src = self._run_tool_search(messages, tools)
            return text, graph_data, src
