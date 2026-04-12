"""
P5 — ReAct Agent with RAG tools
Pattern: Thought -> Action -> Observation -> Answer
Tools: search_documents, search_in_laws, search_by_doc_type, calculate
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv("/opt/rag-experiments/.env")

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import FunctionAgent
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import Settings
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from qdrant_client import QdrantClient

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "p3_multidoc"

def init():
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    client = QdrantClient(url=QDRANT_URL)
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )
    return index

def build_tools(index):

    def search_documents(query: str) -> str:
        """
        Search across all documents in the knowledge base.
        Use this tool for general questions about UDO Ukraine:
        powers, responsibilities, procedures, regulations.
        Input: search query in Ukrainian or English.
        Returns: relevant text passages from all documents.
        """
        query_engine = index.as_query_engine(similarity_top_k=6)
        response = query_engine.query(query)
        sources = []
        for node in response.source_nodes:
            fname = node.metadata.get("file_name", "unknown")
            score = round(node.score, 3) if node.score else "N/A"
            sources.append("[" + fname + ", score:" + str(score) + "]")
        return str(response) + "\n\nSources: " + ", ".join(sources)

    def search_in_laws(query: str) -> str:
        """
        Search specifically in Ukrainian laws about UDO Ukraine.
        Use this tool when you need specific facts: numbers, personnel count,
        quotas, establishment dates, rank limits, total headcount.
        Example queries: total personnel, chyselnist, kilkist osib.
        Input: search query in Ukrainian.
        Returns: text from laws only.
        """
        filters = MetadataFilters(filters=[
            MetadataFilter(key="doc_type", value="law", operator=FilterOperator.EQ)
        ])
        query_engine = index.as_query_engine(similarity_top_k=6, filters=filters)
        response = query_engine.query(query)
        sources = []
        for node in response.source_nodes:
            fname = node.metadata.get("file_name", "unknown")
            score = round(node.score, 3) if node.score else "N/A"
            sources.append("[" + fname + ", score:" + str(score) + "]")
        return str(response) + "\n\nSources: " + ", ".join(sources)

    def search_by_doc_type(query: str, doc_type: str) -> str:
        """
        Search within a specific document type only.
        doc_type options: law, order, decree, general
        Use when you know exactly which document type contains the answer.
        Input: query string and doc_type string.
        Returns: relevant text passages filtered by document type.
        """
        filters = MetadataFilters(filters=[
            MetadataFilter(key="doc_type", value=doc_type, operator=FilterOperator.EQ)
        ])
        query_engine = index.as_query_engine(similarity_top_k=6, filters=filters)
        response = query_engine.query(query)
        return str(response) + "\n[Filtered by doc_type=" + doc_type + "]"

    def calculate(expression: str) -> str:
        """
        Evaluate a simple mathematical expression.
        Use this tool when you need to perform calculations.
        Input: a valid Python math expression, e.g. 2993 * 25000 or 100 / 4.
        Returns: the result of the calculation.
        """
        try:
            allowed = set("0123456789+-*/()., ")
            if not all(c in allowed for c in expression):
                return "Error: only basic math operations allowed"
            result = eval(expression)
            return expression + " = " + str(result)
        except Exception as e:
            return "Calculation error: " + str(e)

    return [
        FunctionTool.from_defaults(fn=search_documents),
        FunctionTool.from_defaults(fn=search_in_laws),
        FunctionTool.from_defaults(fn=search_by_doc_type),
        FunctionTool.from_defaults(fn=calculate),
    ]

def build_agent(index):
    tools = build_tools(index)
    agent = FunctionAgent(
        tools=tools,
        llm=Settings.llm,
        verbose=True,
        allow_parallel_tool_calls=False,
        system_prompt=(
            "You are a helpful assistant that answers questions about UDO Ukraine "
            "(Управління державної охорони України) based on official documents.\n"
            "Always use search tools before answering. Never answer from memory.\n"
            "Rules:\n"
            "- For specific numbers, personnel count, quotas, dates — use search_in_laws first\n"
            "- For general questions about powers, procedures — use search_documents\n"
            "- For calculations — use calculate after finding the numbers\n"
            "- If search_documents finds nothing useful, try search_in_laws\n"
            "Respond in Ukrainian."
        ),
    )
    return agent

async def main():
    print("=" * 60)
    print("P5 — ReAct Agent")
    print("=" * 60)
    print("\nInitializing...")
    index = init()
    agent = build_agent(index)
    print("\nReady! Type your question (or exit to quit)\n")

    while True:
        user_input = input("Question: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Bye.")
            break
        print()
        try:
            response = await agent.run(user_input)
            print("\n" + "=" * 60)
            print("Final Answer: " + str(response))
            print("=" * 60)
        except Exception as e:
            print("Error: " + str(e))
        print()

if __name__ == "__main__":
    asyncio.run(main())
