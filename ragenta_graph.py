import operator
from typing import Annotated, Dict, List, Tuple, TypedDict, Any, Union
import re
import numpy as np
import logging
from tqdm import tqdm

from langgraph.graph import StateGraph, END
from llm_agents import BaseLLMAgent

logger = logging.getLogger("RAGENTA_GRAPH")

class RAGentAState(TypedDict):
    """The state of the RAGentA graph."""
    query: str
    retriever: Any
    agents: Dict[str, BaseLLMAgent]
    n_factor: float
    
    # Internal state
    retrieved_docs: List[Tuple[str, str]]
    doc_answers: List[Tuple[str, str, str]]
    doc_scores: List[float]
    filtered_docs: List[Tuple[str, str]]
    
    # Results
    answer_with_citations: str
    claims: List[Dict[str, Any]]
    
    # Debug/Info
    tau_q: float
    adjusted_tau_q: float
    supporting_passages: List[Tuple[str, str]]
    agent3_prompt: str
    
    # Analysis results
    claim_analysis: str
    question_structure: str
    question_components: List[str]
    unanswered_components: List[str]
    follow_up_questions: List[str]
    coverage_assessment: Dict[str, str]
    claims_to_remove: List[int]
    
    # Final outputs
    final_answer: str
    completely_answered: bool
    follow_up_answers: List[Dict[str, str]]
    
    # Iteration tracking
    iteration: int
    excluded_ids: Annotated[set, operator.or_]

def retrieve_node(state: RAGentAState) -> Dict[str, Any]:
    """Retrieve initial documents."""
    logger.info(f"Retrieving documents for query: {state['query']}")
    docs = state['retriever'].retrieve(state['query'], top_k=20)
    return {"retrieved_docs": docs, "excluded_ids": {doc_id for _, doc_id in docs}}

def score_docs_node(state: RAGentAState) -> Dict[str, Any]:
    """Agent-1 generates initial answers and Agent-2 scores them."""
    query = state['query']
    agent1 = state['agents']['agent1']
    agent2 = state['agents']['agent2']
    
    doc_answers = []
    logger.info("Agent-1 generating answers for each document...")
    for doc_text, doc_id in tqdm(state['retrieved_docs']):
        prompt = f"Document: {doc_text}\nQuestion: {query}\nAnswer:"
        answer = agent1.generate(prompt)
        doc_answers.append((doc_text, doc_id, answer))
        
    logger.info("Agent-2 evaluating and scoring documents...")
    scores = []
    for doc_text, doc_id, answer in tqdm(doc_answers):
        prompt = f"Document: {doc_text}\nQuestion: {query}\nLLM Answer: {answer}\nIs this document relevant and supportive?"
        log_probs = agent2.get_log_probs(prompt, ["Yes", "No"])
        score = log_probs.get("Yes", -100.0) - log_probs.get("No", -100.0)
        scores.append(score)
        
    return {"doc_answers": doc_answers, "doc_scores": scores}

def filter_docs_node(state: RAGentAState) -> Dict[str, Any]:
    """Filter and rank documents based on scores."""
    scores = state['doc_scores']
    n = state['n_factor']
    doc_answers = state['doc_answers']
    
    tau_q = np.mean(scores)
    sigma = np.std(scores)
    adjusted_tau_q = tau_q - n * sigma
    
    filtered_docs = []
    for i, (doc_text, doc_id, _) in enumerate(doc_answers):
        if scores[i] >= adjusted_tau_q:
            filtered_docs.append((doc_text, doc_id, scores[i]))
            
    filtered_docs.sort(key=lambda x: x[2], reverse=True)
    filtered_docs_no_score = [(doc_text, doc_id) for doc_text, doc_id, _ in filtered_docs]
    
    # Fallback if none pass
    if not filtered_docs_no_score:
        filtered_docs_no_score = [(doc_text, doc_id) for doc_text, doc_id, _ in doc_answers]
        
    return {
        "filtered_docs": filtered_docs_no_score,
        "tau_q": tau_q,
        "adjusted_tau_q": adjusted_tau_q
    }

def generate_answer_node(state: RAGentAState) -> Dict[str, Any]:
    """Agent-3 generates final answer with citations."""
    agent3 = state['agents']['agent3']
    docs_text = "\n\n".join([f"Document {i+1}: {text}" for i, (text, _) in enumerate(state['filtered_docs'])])
    
    prompt = f"""Answer based ONLY on documents. Use [X] for citations.
Documents:
{docs_text}
Question: {state['query']}
Answer:"""
    
    answer = agent3.generate(prompt)
    if not answer.strip():
        answer = "I don't have enough information to provide a specific answer."
    
    # Simple claim extraction (mimicking _extract_claims_with_citations)
    bracket_pattern = r"(.*?)\s*\[(\d+(?:,\s*\d+)*)\]"
    matches = re.finditer(bracket_pattern, answer)
    claims = []
    for match in matches:
        claim_text = match.group(1).strip()
        citations = [int(c.strip()) for c in match.group(2).split(",")]
        if claim_text:
            claims.append({"text": claim_text, "citations": citations})
            
    # Sort supporting passages by citation count
    doc_citation_counts = {}
    for claim in claims:
        for doc_idx in claim["citations"]:
            if doc_idx <= len(state['filtered_docs']):
                doc_id = state['filtered_docs'][doc_idx - 1][1]
                doc_citation_counts[doc_id] = doc_citation_counts.get(doc_id, 0) + 1
                
    sorted_doc_ids = sorted(doc_citation_counts.keys(), key=lambda x: doc_citation_counts[x], reverse=True)
    doc_id_map = {doc_id: (doc_text, doc_id) for doc_text, doc_id in state['filtered_docs']}
    supporting_passages = [doc_id_map[doc_id] for doc_id in sorted_doc_ids if doc_id in doc_id_map]
    for doc in state['filtered_docs']:
        if doc[1] not in doc_citation_counts and doc not in supporting_passages:
            supporting_passages.append(doc)
            
    return {
        "answer_with_citations": answer, 
        "claims": claims, 
        "agent3_prompt": prompt,
        "supporting_passages": supporting_passages
    }

def analyze_answer_node(state: RAGentAState) -> Dict[str, Any]:
    """Agent-4 analyzes the answer for completeness and gaps."""
    agent4 = state['agents']['agent4']
    
    docs_text = "\n\n".join([f"Document {i+1}: {text}" for i, (text, _) in enumerate(state['filtered_docs'])])
    claims_text = "\n\n".join([f"Claim {i+1}: {c['text']} [Citations: {c['citations']}]" for i, c in enumerate(state['claims'])])
    
    prompt = f"""Analyze if the question is fully answered.
Question: {state['query']}
Answer: {state['answer_with_citations']}
Claims: {claims_text}
Documents: {docs_text}

Format:
COMPLETELY_ANSWERED: Yes/No
UNANSWERED_ASPECTS: [List]
FOLLOW_UP_QUESTIONS: [List questions starting with '- ']"""

    analysis = agent4.generate(prompt)
    
    comp_match = re.search(r"COMPLETELY_ANSWERED:\s*(Yes|No)", analysis, re.I)
    completely_answered = comp_match.group(1).lower() == "yes" if comp_match else False
    
    follow_ups = []
    fu_section = re.search(r"FOLLOW_UP_QUESTIONS:(.*)", analysis, re.S)
    if fu_section:
        for line in fu_section.group(1).strip().split("\n"):
            match = re.search(r"^-\s*(.*)", line.strip())
            if match:
                follow_ups.append(match.group(1).strip())
                
    return {
        "claim_analysis": analysis,
        "completely_answered": completely_answered,
        "follow_up_questions": follow_ups,
        "final_answer": state['answer_with_citations'] if not state.get('final_answer') else state['final_answer']
    }

def answer_follow_ups_node(state: RAGentAState) -> Dict[str, Any]:
    """Process follow-up questions and integrate them."""
    agent_gen = state['agents']['agent1'] # Use any generation agent
    retriever = state['retriever']
    current_answer = state['final_answer']
    new_follow_up_answers = state.get('follow_up_answers', [])
    excluded_ids = state['excluded_ids']
    
    for fu_q in state['follow_up_questions']:
        logger.info(f"Answering follow-up: {fu_q}")
        new_docs = retriever.retrieve(fu_q, top_k=5, exclude_ids=excluded_ids)
        if not new_docs: continue
        
        excluded_ids.update({doc_id for _, doc_id in new_docs})
        
        docs_text = "\n\n".join([f"Document {i+1}: {text}" for i, (text, _) in enumerate(new_docs)])
        prompt = f"Question: {fu_q}\nDocuments:\n{docs_text}\nAnswer with [X] citations:"
        fu_answer = agent_gen.generate(prompt)
        
        # Integration
        int_prompt = f"Integrate this new info into the previous answer.\nPrev: {current_answer}\nNew: {fu_answer}\nIntegrated:"
        current_answer = agent_gen.generate(int_prompt)
        
        new_follow_up_answers.append({"question": fu_q, "answer": fu_answer})
        
    return {
        "final_answer": current_answer,
        "follow_up_answers": new_follow_up_answers,
        "excluded_ids": excluded_ids,
        "iteration": state.get('iteration', 0) + 1
    }

def should_continue(state: RAGentAState):
    """Decide whether to end or continue with follow-ups."""
    if state['completely_answered'] or not state['follow_up_questions'] or state.get('iteration', 0) >= 3:
        return "end"
    return "continue"

def create_ragenta_graph():
    """Build and compile the RAGentA graph."""
    workflow = StateGraph(RAGentAState)
    
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("score_docs", score_docs_node)
    workflow.add_node("filter_docs", filter_docs_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("analyze_answer", analyze_answer_node)
    workflow.add_node("answer_follow_ups", answer_follow_ups_node)
    
    workflow.set_entry_point("retrieve")
    
    workflow.add_edge("retrieve", "score_docs")
    workflow.add_edge("score_docs", "filter_docs")
    workflow.add_edge("filter_docs", "generate_answer")
    workflow.add_edge("generate_answer", "analyze_answer")
    
    workflow.add_conditional_edges(
        "analyze_answer",
        should_continue,
        {
            "continue": "answer_follow_ups",
            "end": END
        }
    )
    
    workflow.add_edge("answer_follow_ups", "analyze_answer")
    
    return workflow.compile()
