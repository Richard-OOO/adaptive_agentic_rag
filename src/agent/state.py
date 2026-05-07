import operator
from typing import Annotated, List, Dict, Any, Literal
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.documents import Document

class GraphState(TypedDict):
   
    messages: Annotated[list[BaseMessage], add_messages]
    
    question: str
    
    generation: str
    
    search_queries: List[str]  
    
    sql_filters: Dict[str, Any] 

    requires_extended_context: bool 

    need_retrieval: bool 
    
    need_query_optimization: bool
    
    documents: List[Document]  
    
    web_documents: List[Document] 
    
    retrieval_grade: Literal["yes", "no", "partial", "none"]

    past_search_queries: Annotated[List[str], operator.add]
    
    retrieve_loop_step: Annotated[int, operator.add]
    
    web_search_loop_step: Annotated[int, operator.add]
    
    potential_hallucination: bool
    
    matched_domain: str

