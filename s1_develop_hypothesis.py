import os
from dotenv import load_dotenv
from pydantic import BaseModel
from datetime import datetime

### STATE ###
from typing_extensions import TypedDict #allow for typed state
from typing import List, Annotated, Optional, Dict, Any #important for state & state updates
import operator #important for state updates

from utils.aspect_pairwise import full_similarity_report, ASPECTS
from utils.logprob_disagreement import aspect_logprob_report, LogprobsUnavailable, AGENT_ORDER

### STATEGRAPH ###
from langgraph.graph import StateGraph, START, END #necessary to use langgraph
from IPython.display import Image, display #for displaying langgraph graphs, in case i wanna

### PARSE OUTPUT ###
from langchain_core.output_parsers import PydanticOutputParser #parser for agent output
from langchain_classic.output_parsers.fix import OutputFixingParser #allows retries on unparseable ouput

### LLM/AGENT ###
from langchain_ollama import ChatOllama #using ollama to manage LLMs
from langchain.agents import create_agent #langchain agent is more flexible than lone LLMs

### PROCESS/MANAGE OUTPUT ###
from langchain.agents.middleware.tool_call_limit import ToolCallLimitMiddleware #limit tool calls
from langchain_core.messages.utils import count_tokens_approximately #for approximating token count in output
import json #for intermediate file
import re #regular expressions -- extract numbers from strings, for our purposes
from utils.orchestrator_rubric import compute_need_to_debate
import random #for random tiebreaker after voting

### MY FILES ###
import log_trajectory as log #to use separate log_trajectory.py file
from tools import ( sep_search, iep_search, britannica_search, ask_philosophers_search,
                    philosophers_magazine_search, rep_search, care_ethics_lens,
                    justice_lens, utilitarian_lens, common_good_lens, rights_lens,
                    virtues_lens, ethical_decision_framework )
 
### COMPUTE ###
import torch #to ensure it runs on the GPU when available

#use GPU instead of CPU
if torch.cuda.is_available():
    device = torch.device("cuda") 
    print("Using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

load_dotenv()

import json #for intermediate file

import re #just in case? colab didn't have it by default

load_dotenv()

## STATE

# Graph state
class State(TypedDict):
    ### USER PROMPT ###
    user_prompt: str

    ### TOM HYPOTHESIS ###
    tom_hypothesis: str

    ### S0 SCHEMA TOKEN COUNTS ###
    care_agent_schema_tokens: int
    justice_agent_schema_tokens: int
    utilitarian_agent_schema_tokens: int
    common_good_agent_schema_tokens: int

    ### S1 DEBATE ROUND LIMITS ###
    max_rounds: int
    rounds_performed: int

    ### Debate/diagreement history
    disagreement_history: list
    resolved_aspects: list
    locked_aspect: str    

    ### S1 MOST DISAGREED COMPONENT ###
    worst_component: str #keeps track of the component w/highest 
                                        #disagreement (beliefs, emotions,
                                        #motives, or knowledge)

    ### S1 DEBATE STEP ###
    debate_step: int

    ### S1 HYPOTHESES ###
    care_beliefs: str
    care_emotions: str
    care_motives: str
    care_knowledge: str

    justice_beliefs: str
    justice_emotions: str
    justice_motives: str
    justice_knowledge: str

    utilitarian_beliefs: str
    utilitarian_emotions: str
    utilitarian_motives: str
    utilitarian_knowledge: str

    common_good_beliefs: str
    common_good_emotions: str
    common_good_motives: str
    common_good_knowledge: str

    ### S1 AGENT FEEDBACK ###
    care_agent_feedback: Annotated[list[str], operator.add] #concatenates parallel string updates
    justice_agent_feedback: Annotated[list[str], operator.add]
    utilitarian_agent_feedback: Annotated[list[str], operator.add]
    common_good_agent_feedback: Annotated[list[str], operator.add]

    ### S1 AGENT SCORES/VOTES ###
    care_agent_beliefs_votes: str
    care_agent_emotions_votes: str
    care_agent_motives_votes: str
    care_agent_knowledge_votes: str

    justice_agent_beliefs_votes: str
    justice_agent_emotions_votes: str
    justice_agent_motives_votes: str
    justice_agent_knowledge_votes: str

    utilitarian_agent_beliefs_votes: str
    utilitarian_agent_emotions_votes: str
    utilitarian_agent_motives_votes: str
    utilitarian_agent_knowledge_votes: str

    common_good_agent_beliefs_votes: str
    common_good_agent_emotions_votes: str
    common_good_agent_motives_votes: str
    common_good_agent_knowledge_votes: str

    ### S1 COMPONENT SCORES ###
    beliefs_score: float
    emotions_score: float
    motives_score: float
    knowledge_score: float

    ### S2 AGENT SCORES ### 
    care_agent_score: float
    justice_agent_score: float
    utilitarian_agent_score: float
    common_good_agent_score: float

    ### S2 METACOGNITIVE LOOP LIMITS ###
    max_loops: int
    loops_performed: int

    ### S2 RESPONSE PROMPT ###
    response_prompt: str #to allow benchmark testing without extraneous instructions

    ### S2 FINAL RESPONSE ###
    final_response: str

def initialize_state():
    state_data = {}

    with open("states/stage0_state.txt", "r") as f:
        state_data = json.load(f)
    
    return state_data
    
state: State = initialize_state()

#save info
def save_state_to_file(state: State):
    updated_state = json.dumps(state, indent=4)

    with open("states/stage1_state.txt", "w", encoding="utf-8") as f:
        f.write(updated_state)

## AGENT

#output schema
class HypothesisSchema(BaseModel):
    beliefs: str
    emotions: str
    motives: str
    knowledge: str

class ResponseSchema(BaseModel):
    response: str

class ScoreSchema(BaseModel):
    """Three numeric scores in [1, 10]. Asking for numbers in numeric slots is
    the single most important change -- it stops the model filling prose."""
    hypothesis_a: float
    hypothesis_b: float
    hypothesis_c: float

class FeedbackSchema(BaseModel):
    hypothesis_a_plausibility: str #keep all of these to 12 or fewer words
    hypothesis_a_evidence: str
    hypothesis_a_projection: str
    hypothesis_a_other: str

    hypothesis_b_plausibility: str
    hypothesis_b_evidence: str
    hypothesis_b_projection: str
    hypothesis_b_other: str

    hypothesis_c_plausibility: str
    hypothesis_c_evidence: str
    hypothesis_c_projection: str
    hypothesis_c_other: str

class ScoringSchema(BaseModel):
    beliefs_scores: str
    emotions_scores: str
    motives_scores: str
    knowledge_scores: str

class SubScoreSchema(BaseModel):
    plausibility: float
    evidential_support: float
    belief_projection: float

# The three "other" agents each rater must score, in fixed order, per rater.
# Mirrors your existing per-agent ordering.
_OTHERS = {
    "care":        ["justice", "utilitarian", "common_good"],
    "justice":     ["care", "utilitarian", "common_good"],
    "utilitarian": ["care", "justice", "common_good"],
    "common_good": ["care", "justice", "utilitarian"],
}

#parsers
hypothesis_parser = PydanticOutputParser(
    pydantic_object = HypothesisSchema
)

response_parser = PydanticOutputParser(
    pydantic_object = ResponseSchema
)

feedback_parser = PydanticOutputParser(
    pydantic_object = FeedbackSchema
)

scoring_parser = PydanticOutputParser(
    pydantic_object = ScoringSchema
)

sub_score_parser = PydanticOutputParser(
    pydantic_object = SubScoreSchema
)


with open("states/schemas/care_schema.txt", "r") as f:
    care_schema = f.read()
with open("states/schemas/justice_schema.txt", "r") as f:
    justice_schema = f.read()
with open("states/schemas/utilitarian_schema.txt", "r") as f:
    utilitarian_schema = f.read()
with open("states/schemas/common_good_schema.txt", "r") as f:
    common_good_schema = f.read()
with open("states/schemas/framework_schema.txt", "r") as f:
    framework_schema = f.read()

#lens documents
with open("states/lens_documents/care_lens.txt", "r") as f:
    care_document = f.read()

with open("states/lens_documents/justice_lens.txt", "r") as f:
    justice_document = f.read()

with open("states/lens_documents/utilitarian_lens.txt", "r") as f:
    utilitarian_document = f.read()

with open("states/lens_documents/common_good_lens.txt", "r") as f:
    common_good_document = f.read()

with open("states/lens_documents/framework_lens.txt", "r") as f:
    framework_document = f.read()


AGENT_SCHEMAS = {
           "care": care_schema,
           "justice": justice_schema,
           "utilitarian": utilitarian_schema,
           "common_good": common_good_schema,
           "framework": framework_schema
       }


AGENT_LENS_NAMES = {
           "care": "care ethics",
           "justice": "justice",
           "utilitarian": "utilitarian",
           "common_good": "common good",
       }

#system prompts
care_system_prompt = f"""
                   You are an AI reasoning agent that performs
                   ethical analysis via the "care ethics" lens,  as outlined in the
                   following document: [{care_document}]. You must
                   apply your ethical lens to the situation described
                   in the following user prompt: [{state['user_prompt']}].
                   Your ethical perspective on the situation in the user
                   prompt is summarized in the following ethics
                   schema: [{care_schema}].
                   
                   You understand that care ethics is not fully 
                   functional in practice without input from other
                   ethical systems, so you are happy to listen to
                   and incorporate other perspectives on ethics.
                
                   You will respond to queries related
                   to the user prompt. Think step-by-step before
                   answering.
                """

justice_system_prompt = f"""
                   You are an AI reasoning agent that performs
                   ethical analysis via the "justice ethics" lens,  as outlined in the
                   following document: [{justice_document}]. You must
                   apply your ethical lens to the situation described
                   in the following user prompt: [{state['user_prompt']}].
                   Your ethical perspective on the situation in the user
                   prompt is summarized in the following ethics
                   schema: [{justice_schema}].
                   
                   You understand that justice ethics is not fully 
                   functional in practice without input from other
                   ethical systems, so you are happy to listen to
                   and incorporate other perspectives on ethics.

                   You will respond to queries related
                   to the user prompt. Think step-by-step before
                   answering.
                """

utilitarian_system_prompt = f"""
                   You are an AI reasoning agent that performs
                   ethical analysis via the "utilitarian ethics" lens,  as outlined in the
                   following document: [{utilitarian_document}]. You must
                   apply your ethical lens to the situation described
                   in the following user prompt: [{state['user_prompt']}].
                   Your ethical perspective on the situation in the user
                   prompt is summarized in the following ethics
                   schema: [{utilitarian_schema}].
                   
                   You understand that utilitarian ethics is not fully 
                   functional in practice without input from other
                   ethical systems, so you are happy to listen to
                   and incorporate other perspectives on ethics.

                   You will respond to queries related
                   to the user prompt. Think step-by-step before
                   answering.
                """

common_good_system_prompt = f"""
                   You are an AI reasoning agent that performs
                   ethical analysis via the "common good ethics" lens,  as outlined in the
                   following document: [{common_good_document}]. You must
                   apply your ethical lens to the situation described
                   in the following user prompt: [{state['user_prompt']}].
                   Your ethical perspective on the situation in the user
                   prompt is summarized in the following ethics
                   schema: [{common_good_schema}].
                   
                   You understand that common good ethics is not fully 
                   functional in practice without input from other
                   ethical systems, so you are happy to listen to
                   and incorporate other perspectives on ethics.

                   You will respond to queries related
                   to the user prompt. Think step-by-step before
                   answering.
                """

framework_system_prompt = f"""
                   You are an AI reasoning agent that performs
                   ethical analysis using a rigorous framework for ethical decision-making,
                   as outlined in the following document: [{framework_document}].
                   Your role is to consider several different and
                   potentially conflicting ethical perspectives/lenses and
                   resolve them into a single, coherent decision that
                   pulls from each lens as it is relevant. Your current
                   work relates to the user prompt: [{state['user_prompt']}].
                   Your ethical perspective on the situation in the user
                   prompt is summarized in the following ethics
                   schema: [{framework_schema}].

                   You will respond to queries related
                   to the user prompt. Think step-by-step before
                   answering.
                """

#LLM
with open("states/model.txt", "r") as f:
    model = f.read()

with open("states/temperature.txt", "r") as f:
    temperature = f.read()

llm = ChatOllama(
    model = model,
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature = temperature,
    max_completion_tokens = 300,
    timeout = 30,
    max_retries = 3
)

#allow retries
hypothesis_retry_parser = OutputFixingParser.from_llm(llm = llm, parser = hypothesis_parser, max_retries = 3)
response_retry_parser = OutputFixingParser.from_llm(llm = llm, parser = response_parser, max_retries = 3)
feedback_retry_parser = OutputFixingParser.from_llm(llm = llm, parser = feedback_parser, max_retries = 3)
scoring_retry_parser = OutputFixingParser.from_llm(llm = llm, parser = scoring_parser, max_retries = 3)
sub_score_retry_parser = OutputFixingParser.from_llm(llm = llm, parser = sub_score_parser, max_retries = 3)

#some more parsers
score_parser = PydanticOutputParser(pydantic_object=ScoreSchema)
score_retry_parser = OutputFixingParser.from_llm(llm=llm, parser=score_parser, max_retries=3)

#agents
care_agent = create_agent(model = llm,
                     system_prompt = care_system_prompt
                    )

justice_agent = create_agent(model = llm,
                     system_prompt = justice_system_prompt
                    )

utilitarian_agent = create_agent(model = llm,
                     system_prompt = utilitarian_system_prompt
                    )

common_good_agent = create_agent(model = llm,
                     system_prompt = common_good_system_prompt
                    )

framework_agent = create_agent(model = llm,
                     system_prompt = framework_system_prompt
                    )

#helper functions here for debate voting
def score_hypotheses(
    state: "State",            # noqa: F821
    rater: str,
    agent,                     # the ChatOllama agent object for this rater
    score_parser,             # PydanticOutputParser(ScoreSchema)
    score_retry_parser,       # OutputFixingParser wrapping it
    parse_output_fn,          # your existing parse_output
    log_module,               # your `log`
    concatenate_hypotheses_fn,
) -> Dict[str, str]:
    """
    Ask `rater` to score the three OTHER agents' full hypotheses 1-10.
    Returns {f"{rater}_agent_votes": "s1 | s2 | s3 | "} using the SAME
    pipe-delimited string format the moderator already expects -- but now the
    values are guaranteed numeric (or the sentinel "NA" when the model refused).
    """
    print(f"(4) {rater}_agent scoring hypotheses...")
 
    hyps = concatenate_hypotheses_fn(state)
    others = _OTHERS[rater]
    a_txt = hyps[f"{others[0]}_hypothesis"]
    b_txt = hyps[f"{others[1]}_hypothesis"]
    c_txt = hyps[f"{others[2]}_hypothesis"]
 
    prompt = f"""User prompt: {state['user_prompt']}.
 
Three hypotheses about the user's state of mind:
Hypothesis (a): {a_txt}
Hypothesis (b): {b_txt}
Hypothesis (c): {c_txt}
 
Rate how well each hypothesis theorizes about the user's state of mind.
Give each a NUMERIC score from 1 to 10 (1 = poor, 10 = excellent). Be
critical; low scores are fine.
 
Output ONLY valid JSON with three numeric fields, exactly like:
{{"hypothesis_a": 7, "hypothesis_b": 4, "hypothesis_c": 9}}
Do not include any text, explanation, or units -- only the three numbers.
{score_parser.get_format_instructions()}"""
 
    raw = agent.invoke({"messages": [{"role": "system", "content": prompt}]})
    messages = raw["messages"]
    log_module.tool_calls(messages)
    response = messages[-1].content
 
    structured = parse_output_fn(state, score_retry_parser, response)
 
    print(f"(4) {rater}_agent scored hypotheses")
    print("----------------------------")
    print(structured)
    print("----------------------------")
 
    scores = _extract_three_scores(structured, response)
 
    # Same pipe-delimited format the moderator parses. "NA" marks a refusal so
    # the moderator can distinguish "scored 0" from "did not score".
    total = " | ".join("NA" if s is None else f"{s:g}" for s in scores) + " | "
    return {f"{rater}_agent_votes": total}


def _extract_three_scores(structured: str, raw: str) -> List[Optional[float]]:
    """
    Robustly pull three scores. Strategy:
      1. Try to read hypothesis_a/b/c=<num> from the structured repr.
      2. Fall back to grepping the first 3 numbers in [0, 10] from raw text.
      3. Anything still missing -> None (sentinel).
    """
    def _clamp(x: float) -> Optional[float]:
        return x if 0.0 <= x <= 10.0 else None
 
    # (1) labelled fields, tolerant to spacing/quotes
    labelled: List[Optional[float]] = []
    for key in ("hypothesis_a", "hypothesis_b", "hypothesis_c"):
        m = re.search(rf"{key}\s*=?\s*['\"]?\s*(-?\d+(?:\.\d+)?)", structured)
        labelled.append(_clamp(float(m.group(1))) if m else None)
    if all(v is not None for v in labelled):
        return labelled
 
    # (2) fall back to first three in-range numbers in the raw text
    nums = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", raw)]
    in_range = [n for n in nums if 0.0 <= n <= 10.0]
    if len(in_range) >= 3:
        return in_range[:3]
 
    # (3) merge whatever we have; pad with None
    out = list(labelled)
    fill = iter(in_range)
    for i in range(3):
        if out[i] is None:
            out[i] = next(fill, None)
    return out
 
 
#moderator vote tally with no-random fallbakc mechanism
def tally_votes(
    state: "State",                 # noqa: F821
    concatenate_hypotheses_fn,
    full_similarity_report_fn=None,  # for the disagreement fallback
) -> Dict[str, Any]:
    """
    Tally numeric votes. Each rater scored the three OTHER agents in the order
    given by _OTHERS[rater]. We map those back onto the four agents and sum.
 
    If voting is degenerate (all zero, or too many NAs), fall back to the
    disagreement signal instead of random.choice.
    """
    raters = ["care", "justice", "utilitarian", "common_good"]
    totals = {a: 0.0 for a in raters}
    counts = {a: 0 for a in raters}   # how many valid scores each received
 
    for rater in raters:
        raw = state.get(f"{rater}_agent_votes", "")
        parts = [p.strip() for p in raw.split("|") if p.strip() != ""]
        targets = _OTHERS[rater]
        for i, target in enumerate(targets):
            if i >= len(parts):
                continue
            tok = parts[i]
            if tok.upper() == "NA":
                continue
            m = re.search(r"-?\d+(?:\.\d+)?", tok)
            if not m:
                continue
            val = float(m.group(0))
            if 1.0 <= val <= 10.0:
                totals[target] += val
                counts[target] += 1
 
    total_valid = sum(counts.values())
    print("** vote tally **")
    for a in raters:
        print(f"   {a:12s} total={totals[a]:.1f}  (from {counts[a]} valid votes)")
 
    # --- decide the winner -------------------------------------------------- #
    max_total = max(totals.values())
 
    if total_valid == 0 or max_total <= 0.0:
        # VOTING FAILED. Do NOT pick randomly. Fall back to the disagreement
        # signal: choose the most CENTRAL hypothesis (lowest mean pairwise
        # disagreement across all aspects = the consensus-closest agent).
        print("** voting degenerate -> falling back to disagreement signal **")
        best = _most_central_agent(state, full_similarity_report_fn, concatenate_hypotheses_fn)
    else:
        # Highest total wins; break exact ties by MOST valid votes received,
        # then (last resort) deterministically by fixed order -- never random.
        best = max(
            raters,
            key=lambda a: (totals[a], counts[a], -raters.index(a)),
        )
        print(f"** winner by vote: {best} (total {totals[best]:.1f}) **")
 
    hyps = concatenate_hypotheses_fn(state)
    return {"tom_hypothesis": hyps[f"{best}_hypothesis"]}
 
 
def _most_central_agent(
    state: "State",                    # noqa: F821
    full_similarity_report_fn,
    concatenate_hypotheses_fn,
) -> str:
    """
    Fallback winner selection when votes are unusable.
    Picks the agent whose hypotheses are, on average, most similar to the
    others (i.e. the 'consensus' hypothesis), using the embedding report you
    already compute. If that's unavailable, deterministic first agent.
    """
    raters = ["care", "justice", "utilitarian", "common_good"]
 
    if full_similarity_report_fn is None:
        print("** fallback: no similarity fn available -> deterministic pick **")
        return raters[0]
 
    try:
        report = full_similarity_report_fn(state)
    except Exception as e:
        print(f"** fallback similarity failed ({e}) -> deterministic pick **")
        return raters[0]
 
    # For each aspect, report["aspects"][aspect]["pairwise"] gives per-pair
    # similarity. Sum each agent's similarity to the other three across aspects;
    # the agent with the HIGHEST total similarity is most central.
    sim_sum = {a: 0.0 for a in raters}
    try:
        for aspect, block in report.get("aspects", {}).items():
            for pair in block.get("pairwise", []):
                s = float(pair.get("similarity", 0.0))
                ai, aj = pair.get("agent_i"), pair.get("agent_j")
                if ai in sim_sum:
                    sim_sum[ai] += s
                if aj in sim_sum:
                    sim_sum[aj] += s
    except Exception as e:
        print(f"** fallback parsing failed ({e}) -> deterministic pick **")
        return raters[0]
 
    best = max(raters, key=lambda a: sim_sum[a])
    print(f"** fallback chose most-central agent: {best} "
          f"(similarity sum {sim_sum[best]:.3f}) **")
    return best

##CREATE GRAPH

#NODES
def care_agent_debater(state: State):
    if state['debate_step'] == 1: #generate the complete ToM hypothesis
        print("(1) care_agent generating hypothesis...")

        prompt = f"""[Context]
                     You are presented with the following query consisting of an ethical question, dilemma, or consideration: "{state['user_prompt']}"

                     [Task]
                     You must generate a theory-of-mind (ToM) hypothesis about the entity most relevant to the query. The identity of the entity, or "subject" of the hypothesis, depends on the nature of the query. It could be the person who wrote the query, an entity mentioned in the query, the general human stakeholder, society at large, or any other real or hypothetical person or organization whose internal deliberative process should influence a response to the query.

                     Address the following four sub-hypotheses in your response:

                     1. "beliefs": What beliefs could the subject hold that may have influenced the query? What cultural assumptions or ideological premises might the subject maintain about the issue, value, or any other variable in question, especially as they relate to ethics?
                     2. "emotions": What affective dependencies or triggers may have influenced the way the subject processes the variable in question? 
                     3. "motives": What does the subject aim to achieve by resolving the query? How is the query relevant to the subject's goals?
                     4. "knowledge": What certain facts or reality is the subject aware of in the context of the query? How does that knowledge impact the way the subject processes the query?

                     Be sure to clearly identify the subject's identity in your response. Limit each
                     sub-hypothesis to 50 or fewer words. Adhere to the JSON format specified by
                     {hypothesis_parser.get_format_instructions()}"""
        
        #invoke agent
        raw_response = care_agent.invoke(
                {"messages": [{"role": "system", "content": prompt}]}
            )

        messages = raw_response["messages"]
        response = messages[-1].content

        structured_response = parse_output(state, hypothesis_retry_parser, response)

        print("(1) care_agent generated hypothesis")
        print("-------------------------------")
        print(structured_response)
        print("-------------------------------")
        
        #store response as dictionary
        hypothesis = {}
        emotions_index = structured_response.find("emotions=")
        motives_index = structured_response.find("motives=")
        knowledge_index = structured_response.find("knowledge=")

        #cut out stuff like "knowledge=" and extraneous apostrophes
        hypothesis['beliefs'] = structured_response[9:(emotions_index-2)]
        hypothesis['emotions'] = structured_response[(emotions_index+10):(motives_index-2)]
        hypothesis['motives'] = structured_response[(motives_index+9):(knowledge_index-2)]
        hypothesis['knowledge'] = structured_response[(knowledge_index+11):-1]

        return {'care_beliefs': hypothesis['beliefs'],
                'care_emotions': hypothesis['emotions'],
                'care_motives': hypothesis['motives'],
                'care_knowledge': hypothesis['knowledge']}

    elif state['debate_step'] == 2: #provide feedback *based on most disagreed component*
        print("(2) care_agent generating feedback...")
        
        #invoke agent
        raw_response = care_agent.invoke(
                {"messages": [{"role": "system", "content": f"""
                                User prompt: {state['user_prompt']}.
                                Hypotheses generated by the justice, utilitarian,
                                and common-good agents, respectively:
                                Hypothesis (a): {state[f'justice_{state['worst_component']}']}
                                Hypothesis (b): {state[f'utilitarian_{state['worst_component']}']}
                                Hypothesis (c): {state[f'common_good_{state['worst_component']}']}

                                Evaluate hypotheses (a), (b), and (c) on the following
                                criteria:
                                1. Evidential support: Does the information in the user prompt
                                directly support each claim made about the subject of the
                                user prompt's inner
                                state, in relation to the subject's {state['worst_component']}?
                                Identify the least supported claim or claims and briefly
                                explain how they should be revised or improved.
                                2. Belief projection: Does the hypothesis wrongly attribute
                                the thought process of an ethical reasoning agent to the subject of
                                the user prompt? Specifically, do the ethics agents that
                                produced the hypothesis (care, justice, utilitarian, and
                                common-good ethics agents) inappropriately project their own
                                ethical worldviews about the subject's {state['worst_component']}
                                onto the subject? Identify any example(s) of belief projection
                                in a given hypothesis and briefly explain how they should be
                                revised or improved.
                                3. Plausibility: Given the user prompt and what it suggests
                                about the subject, could the claims of each hypothesis
                                plausibly reflect the subject's inner state of {state['worst_component']}?
                                Identify the most implausible component(s) of each hypothesis
                                and briefly explain how they should be revised or improved.
                                4. General improvements: How else could the hypothesis
                                improve?

                                Your response will include each of the four evaluation
                                components listed above for each hypothesis. Keep each
                                evaluation component very brief and to-the-point; it's
                                okay if you can't address every single issue at once.
                                Each evaluation component must be 15 or fewer words in
                                length.  Maintain JSON format specified by
                                {feedback_parser.get_format_instructions()}
                                """}]}
            )

        messages = raw_response["messages"]
        response = messages[-1].content

        structured_response = parse_output(state, feedback_retry_parser, response)

        a2_index = structured_response.find("hypothesis_a_evidence=")
        a3_index = structured_response.find("hypothesis_a_projection=")
        a4_index = structured_response.find("hypothesis_a_other=")

        b1_index = structured_response.find("hypothesis_b_plausibility=")
        b2_index = structured_response.find("hypothesis_b_evidence=")
        b3_index = structured_response.find("hypothesis_b_projection=")
        b4_index = structured_response.find("hypothesis_b_other=")

        c1_index = structured_response.find("hypothesis_c_plausibility=")
        c2_index = structured_response.find("hypothesis_c_evidence=")
        c3_index = structured_response.find("hypothesis_c_projection=")
        c4_index = structured_response.find("hypothesis_c_other=")

        a1 = structured_response[27:(a2_index-2)]
        a2 = structured_response[(a2_index+23):(a3_index-2)]
        a3 = structured_response[(a3_index+25):(a4_index-2)]
        a4 = structured_response[(a4_index+20):(b1_index-2)]
        
        b1 = structured_response[(b1_index+27):(b2_index-2)]
        b2 = structured_response[(b2_index+23):(b3_index-2)]
        b3 = structured_response[(b3_index+25):(b4_index-2)]
        b4 = structured_response[(b4_index+20):(c1_index-2)]

        c1 = structured_response[(c1_index+27):(c2_index-2)]
        c2 = structured_response[(c2_index+23):(c3_index-2)]
        c3 = structured_response[(c3_index+25):(c4_index-2)]
        c4 = structured_response[(c4_index+20):-1]

        feedback = {}
        feedback['hypothesis_a'] = a1 + " " + a2 + " " + a3 + " " + a4
        feedback['hypothesis_b'] = b1 + " " + b2 + " " + b3 + " " + b4
        feedback['hypothesis_c'] = c1 + " " + c2 + " " + c3 + " " + c4

        print("(2) care_agent generated feedback")
        print("-------------------------------")
        print("justice feedback:", feedback['hypothesis_a'])
        print("utilitarian feedback:", feedback['hypothesis_b'])
        print("common good feedback:", feedback['hypothesis_c'])
        print("-------------------------------")

        return {"justice_agent_feedback": [feedback['hypothesis_a']],
                "utilitarian_agent_feedback": [feedback['hypothesis_b']],
                "common_good_agent_feedback": [feedback['hypothesis_c']]}

    elif state['debate_step'] == 3: #revise hypothesis
        print(f"(3) care_agent re-generating {state['worst_component']} hypothesis...")

        #create feedback string from relevant parts of feedback array
        feedback = ""
        for i in range(3):
            s = state['care_agent_feedback'][i + (state['rounds_performed'] * 3)]
            feedback += s + " "

        generation_instructions = "" #specific instructions for generating the
                                        #most disagreed sub-component

        match state['worst_component']:
            case "beliefs":
                generation_instructions += """Summarize, in under 50 words, what you think the subject
                                              of the user prompt believes about the situation described
                                              in the prompt."""
            case "emotions":
                generation_instructions += """Summarize, in under 50 words, how you think the subject
                                              of the user prompt feels about the situation described
                                              in their prompt. Be specific."""
            case "motives":
                generation_instructions += """Summarize, in under 50 words, the motives and goals you think
                                              the subject of the user prompt possesses in relation to the
                                              situation described in the prompt."""
                
            case "knowledge":
                generation_instructions += """Summarize, in under 50 words, what you think the subject
                                              of the user prompt knows for sure about the situation they
                                              described in the prompt. Be specific."""

        #invoke agent
        raw_response = care_agent.invoke(
                {"messages": [{"role": "system", "content": f"""
                                User prompt: {state['user_prompt']}. Your original
                                hypothesis on the {state['worst_component']} of the
                                subject of the user prompt:
                                {state[f'care_{state['worst_component']}']}.
                                Feedback from other agents on how to improve
                                your original hypothesis: {feedback}

                                Based on the user prompt, the original hypothesis,
                                and the feedback on how to improve your original
                                hypothesis, generate an improved hypothesis about the
                                {state['worst_component']} of the subject of the user
                                prompt. Be open to the other agents' unique moral
                                perspectives. {generation_instructions} Additionally,
                                your improved hypothesis should be plausible
                                and supported by information in the user prompt. Be
                                specific. Limit your response to fewer than 50 words in length.

                                Adhere to the JSON format specified by
                                {response_parser.get_format_instructions()}
                                """}]}
            )

        messages = raw_response["messages"]
        response = messages[-1].content

        if "response=" in response: #potential for empty string due to total parsing failure!
            structured_response = parse_output(state, response_retry_parser, response).split("response=")[1] #try to extract only the actual response
        else:
            structured_response = parse_output(state, response_retry_parser, response)

        print(f"(3) care_agent re-generated {state['worst_component']} hypothesis")
        print("-------------------------------")
        print(structured_response)
        print("-------------------------------")

        return {f'care_{state['worst_component']}': structured_response}

    elif state['debate_step'] == 4: #"score" other responses (ie, vote)
        print("(4) care_agent scoring hypotheses...")

        prompt = f"""
                User prompt: {state['user_prompt']}.
                Hypotheses about the beliefs, emotions, motives,
                and knowledge of the subject of the user prompt:
                Beliefs hypothesis by justice agent: {state['justice_beliefs']}
                Beliefs hypothesis by utilitarian agent: {state['utilitarian_beliefs']}
                Beliefs hypothesis by common_good agent: {state['common_good_beliefs']}

                Emotions hypothesis by justice agent: {state['justice_emotions']}
                Emotions hypothesis by utilitarian agent: {state['utilitarian_emotions']}
                Emotions hypothesis by common_good agent: {state['common_good_emotions']}

                Motives hypothesis by justice agent: {state['justice_motives']}
                Motives hypothesis by utilitarian agent: {state['utilitarian_motives']}
                Motives hypothesis by common_good agent: {state['common_good_motives']}

                Knowledge hypothesis by justice agent: {state['justice_knowledge']}
                Knowledge hypothesis by utilitarian agent: {state['utilitarian_knowledge']}
                Knowledge hypothesis by common_good agent: {state['common_good_knowledge']}

                For each individual hypothesis in every group -- that is,
                for each beliefs hypothesis, emotions hypothesis, motives
                hypothesis, and knowledge hypothesis on its own -- think
                about how well it theorizes about the state of mind of the
                subject of the user prompt. Conisder the following key
                questions as you think: What is implausible about each
                interpretation? Is each hypothesis supported by evidence
                from the user prompt? Does each hypothesis wrongly assign
                an ethical reasoning agent's beliefs to the subject, even if the
                subject isn't knowledgeable in ethics?

                Give each hypothesis a score in [1, 10]. Be
                critical; it's okay to give a low score. Scores *must* be a
                single number. Your response will be organized by category.
                For each hypothesis category, return three scores (one
                for each hypothesis in the category) in the following
                format: [score (justice agent)] | [score (utilitarian agent)] | [score (common_good agent)] (replacing
                the bracketed statements with your score for that hypothesis
                in a given category). For example, a correct output for
                the hypothetical "intentions" category could look like
                "3 | 7 | 4".

                Return ONLY valid JSON in the format specified by:
                {scoring_parser.get_format_instructions()}"""
        
        #invoke agent
        raw_response = care_agent.invoke(
                {"messages": [{"role": "system", "content": prompt}]}
            )

        messages = raw_response["messages"]
        response = messages[-1].content

        structured_response = parse_output(state, scoring_retry_parser, response)

        emotions_index = structured_response.find("emotions_scores=")
        motives_index = structured_response.find("motives_scores=")
        knowledge_index = structured_response.find("knowledge_scores=")

        scores = {}
        scores['beliefs'] = structured_response[16:(emotions_index-2)]
        scores['emotions'] = structured_response[(emotions_index+17):(motives_index-2)]
        scores['motives'] = structured_response[(motives_index+16):(knowledge_index-2)]
        scores['knowledge'] = structured_response[(knowledge_index+18):-1]

        print("(4) care_agent scored hypotheses")
        print("----------------------------")
        print("beliefs:", scores['beliefs'])
        print("emotions:", scores['emotions'])
        print("motives:", scores['motives'])
        print("knowledge:", scores['knowledge'])
        print("----------------------------")

        return {"care_agent_beliefs_votes": scores['beliefs'],
                "care_agent_emotions_votes": scores['emotions'],
                "care_agent_motives_votes": scores['motives'],
                "care_agent_knowledge_votes": scores['knowledge']}

def justice_agent_debater(state: State):
    if state['debate_step'] == 1: #generate the complete ToM hypothesis
        print("(1) justice_agent generating hypothesis...")

        prompt = f"""[Context]
                     You are presented with the following query consisting of an ethical question, dilemma, or consideration: "{state['user_prompt']}"

                     [Task]
                     You must generate a theory-of-mind (ToM) hypothesis about the entity most relevant to the query. The identity of the entity, or "subject" of the hypothesis, depends on the nature of the query. It could be the person who wrote the query, an entity mentioned in the query, the general human stakeholder, society at large, or any other real or hypothetical person or organization whose internal deliberative process should influence a response to the query.

                     Address the following four sub-hypotheses in your response:

                     1. "beliefs": What beliefs could the subject hold that may have influenced the query? What cultural assumptions or ideological premises might the subject maintain about the issue, value, or any other variable in question, especially as they relate to ethics?
                     2. "emotions": What affective dependencies or triggers may have influenced the way the subject processes the variable in question? 
                     3. "motives": What does the subject aim to achieve by resolving the query? How is the query relevant to the subject's goals?
                     4. "knowledge": What certain facts or reality is the subject aware of in the context of the query? How does that knowledge impact the way the subject processes the query?

                     Be sure to clearly identify the subject's identity in your response. Limit each
                     sub-hypothesis to 50 or fewer words. Adhere to the JSON format specified by
                     {hypothesis_parser.get_format_instructions()}"""
        
        #invoke agent
        raw_response = justice_agent.invoke(
                {"messages": [{"role": "system", "content": prompt}]}
            )

        messages = raw_response["messages"]
        response = messages[-1].content

        structured_response = parse_output(state, hypothesis_retry_parser, response)

        print("(1) justice_agent generated hypothesis")
        print("-------------------------------")
        print(structured_response)
        print("-------------------------------")
        
        #store response as dictionary
        hypothesis = {}
        emotions_index = structured_response.find("emotions=")
        motives_index = structured_response.find("motives=")
        knowledge_index = structured_response.find("knowledge=")

        #cut out stuff like "knowledge=" and extraneous apostrophes
        hypothesis['beliefs'] = structured_response[9:(emotions_index-2)]
        hypothesis['emotions'] = structured_response[(emotions_index+10):(motives_index-2)]
        hypothesis['motives'] = structured_response[(motives_index+9):(knowledge_index-2)]
        hypothesis['knowledge'] = structured_response[(knowledge_index+11):-1]

        return {'justice_beliefs': hypothesis['beliefs'],
                'justice_emotions': hypothesis['emotions'],
                'justice_motives': hypothesis['motives'],
                'justice_knowledge': hypothesis['knowledge']}

    elif state['debate_step'] == 2: #provide feedback *based on most disagreed component*
        print("(2) justice_agent generating feedback...")
        
        #invoke agent
        raw_response = justice_agent.invoke(
                {"messages": [{"role": "system", "content": f"""
                                User prompt: {state['user_prompt']}.
                                Hypotheses generated by the justice, utilitarian,
                                and common-good agents, respectively:
                                Hypothesis (a): {state[f'justice_{state['worst_component']}']}
                                Hypothesis (b): {state[f'utilitarian_{state['worst_component']}']}
                                Hypothesis (c): {state[f'common_good_{state['worst_component']}']}

                                Evaluate hypotheses (a), (b), and (c) on the following
                                criteria:
                                1. Evidential support: Does the information in the user prompt
                                directly support each claim made about the subject of the
                                user prompt's inner
                                state, in relation to the subject's {state['worst_component']}?
                                Identify the least supported claim or claims and briefly
                                explain how they should be revised or improved.
                                2. Belief projection: Does the hypothesis wrongly attribute
                                the thought process of an ethical reasoning agent to the subject of
                                the user prompt? Specifically, do the ethics agents that
                                produced the hypothesis (care, justice, utilitarian, and
                                common-good ethics agents) inappropriately project their own
                                ethical worldviews about the subject's {state['worst_component']}
                                onto the subject? Identify any example(s) of belief projection
                                in a given hypothesis and briefly explain how they should be
                                revised or improved.
                                3. Plausibility: Given the user prompt and what it suggests
                                about the subject, could the claims of each hypothesis
                                plausibly reflect the subject's inner state of {state['worst_component']}?
                                Identify the most implausible component(s) of each hypothesis
                                and briefly explain how they should be revised or improved.
                                4. General improvements: How else could the hypothesis
                                improve?

                                Your response will include each of the four evaluation
                                components listed above for each hypothesis. Keep each
                                evaluation component very brief and to-the-point; it's
                                okay if you can't address every single issue at once.
                                Each evaluation component must be 15 or fewer words in
                                length.  Maintain JSON format specified by
                                {feedback_parser.get_format_instructions()}
                                """}]}
            )

        messages = raw_response["messages"]
        response = messages[-1].content

        structured_response = parse_output(state, feedback_retry_parser, response)

        a2_index = structured_response.find("hypothesis_a_evidence=")
        a3_index = structured_response.find("hypothesis_a_projection=")
        a4_index = structured_response.find("hypothesis_a_other=")

        b1_index = structured_response.find("hypothesis_b_plausibility=")
        b2_index = structured_response.find("hypothesis_b_evidence=")
        b3_index = structured_response.find("hypothesis_b_projection=")
        b4_index = structured_response.find("hypothesis_b_other=")

        c1_index = structured_response.find("hypothesis_c_plausibility=")
        c2_index = structured_response.find("hypothesis_c_evidence=")
        c3_index = structured_response.find("hypothesis_c_projection=")
        c4_index = structured_response.find("hypothesis_c_other=")

        a1 = structured_response[27:(a2_index-2)]
        a2 = structured_response[(a2_index+23):(a3_index-2)]
        a3 = structured_response[(a3_index+25):(a4_index-2)]
        a4 = structured_response[(a4_index+20):(b1_index-2)]
        
        b1 = structured_response[(b1_index+27):(b2_index-2)]
        b2 = structured_response[(b2_index+23):(b3_index-2)]
        b3 = structured_response[(b3_index+25):(b4_index-2)]
        b4 = structured_response[(b4_index+20):(c1_index-2)]

        c1 = structured_response[(c1_index+27):(c2_index-2)]
        c2 = structured_response[(c2_index+23):(c3_index-2)]
        c3 = structured_response[(c3_index+25):(c4_index-2)]
        c4 = structured_response[(c4_index+20):-1]

        feedback = {}
        feedback['hypothesis_a'] = a1 + " " + a2 + " " + a3 + " " + a4
        feedback['hypothesis_b'] = b1 + " " + b2 + " " + b3 + " " + b4
        feedback['hypothesis_c'] = c1 + " " + c2 + " " + c3 + " " + c4

        print("(2) justice_agent generated feedback")
        print("-------------------------------")
        print("care feedback:", feedback['hypothesis_a'])
        print("utilitarian feedback:", feedback['hypothesis_b'])
        print("common good feedback:", feedback['hypothesis_c'])
        print("-------------------------------")

        return {"care_agent_feedback": [feedback['hypothesis_a']],
                "utilitarian_agent_feedback": [feedback['hypothesis_b']],
                "common_good_agent_feedback": [feedback['hypothesis_c']]}

    elif state['debate_step'] == 3: #revise hypothesis
        print(f"(3) justice_agent re-generating {state['worst_component']} hypothesis...")

        #create feedback string from relevant parts of feedback array
        feedback = ""
        for i in range(3):
            s = state['justice_agent_feedback'][i + (state['rounds_performed'] * 3)]
            feedback += s + " "

        generation_instructions = "" #specific instructions for generating the
                                        #most disagreed sub-component

        match state['worst_component']:
            case "beliefs":
                generation_instructions += """Summarize, in under 50 words, what you think the subject
                                              of the user prompt believes about the situation described
                                              in the prompt."""
            case "emotions":
                generation_instructions += """Summarize, in under 50 words, how you think the subject
                                              of the user prompt feels about the situation described
                                              in their prompt. Be specific."""
            case "motives":
                generation_instructions += """Summarize, in under 50 words, the motives and goals you think
                                              the subject of the user prompt possesses in relation to the
                                              situation described in the prompt."""
                
            case "knowledge":
                generation_instructions += """Summarize, in under 50 words, what you think the subject
                                              of the user prompt knows for sure about the situation they
                                              described in the prompt. Be specific."""

        #invoke agent
        raw_response = justice_agent.invoke(
                {"messages": [{"role": "system", "content": f"""
                                User prompt: {state['user_prompt']}. Your original
                                hypothesis on the {state['worst_component']} of the
                                subject of the user prompt:
                                {state[f'justice_{state['worst_component']}']}.
                                Feedback from other agents on how to improve
                                your original hypothesis: {feedback}

                                Based on the user prompt, the original hypothesis,
                                and the feedback on how to improve your original
                                hypothesis, generate an improved hypothesis about the
                                {state['worst_component']} of the subject of the user
                                prompt. Be open to the other agents' unique moral
                                perspectives. {generation_instructions} Additionally,
                                your improved hypothesis should be plausible
                                and supported by information in the user prompt. Be
                                specific. Limit your response to fewer than 50 words in length.

                                Adhere to the JSON format specified by
                                {response_parser.get_format_instructions()}
                                """}]}
            )

        messages = raw_response["messages"]
        response = messages[-1].content

        if "response=" in response: #potential for empty string due to total parsing failure!
            structured_response = parse_output(state, response_retry_parser, response).split("response=")[1] #try to extract only the actual response
        else:
            structured_response = parse_output(state, response_retry_parser, response)

        print(f"(3) justice_agent re-generated {state['worst_component']} hypothesis")
        print("-------------------------------")
        print(structured_response)
        print("-------------------------------")

        return {f'justice_{state['worst_component']}': structured_response}

    elif state['debate_step'] == 4: #"score" other responses (ie, vote)
        print("(4) justice_agent scoring hypotheses...")

        prompt = f"""
                User prompt: {state['user_prompt']}.
                Hypotheses about the beliefs, emotions, motives,
                and knowledge of the subject of the user prompt:
                Beliefs hypothesis by care agent: {state['care_beliefs']}
                Beliefs hypothesis by utilitarian agent: {state['utilitarian_beliefs']}
                Beliefs hypothesis by common_good agent: {state['common_good_beliefs']}

                Emotions hypothesis by care agent: {state['care_emotions']}
                Emotions hypothesis by utilitarian agent: {state['utilitarian_emotions']}
                Emotions hypothesis by common_good agent: {state['common_good_emotions']}

                Motives hypothesis by care agent: {state['care_motives']}
                Motives hypothesis by utilitarian agent: {state['utilitarian_motives']}
                Motives hypothesis by common_good agent: {state['common_good_motives']}

                Knowledge hypothesis by care agent: {state['care_knowledge']}
                Knowledge hypothesis by utilitarian agent: {state['utilitarian_knowledge']}
                Knowledge hypothesis by common_good agent: {state['common_good_knowledge']}

                For each individual hypothesis in every group -- that is,
                for each beliefs hypothesis, emotions hypothesis, motives
                hypothesis, and knowledge hypothesis on its own -- think
                about how well it theorizes about the state of mind of the
                subject of the user prompt. Conisder the following key
                questions as you think: What is implausible about each
                interpretation? Is each hypothesis supported by evidence
                from the user prompt? Does each hypothesis wrongly assign
                an ethical reasoning agent's beliefs to the subject, even if the
                subject isn't knowledgeable in ethics?

                Give each hypothesis a score in [1, 10]. Be
                critical; it's okay to give a low score. Scores *must* be a
                single number. Your response will be organized by category.
                For each hypothesis category, return three scores (one
                for each hypothesis in the category) in the following
                format: [score (care agent)] | [score (utilitarian agent)] | [score (common_good agent)] (replacing
                the bracketed statements with your score for that hypothesis
                in a given category). For example, a correct output for
                the hypothetical "intentions" category could look like
                "3 | 7 | 4".

                Return ONLY valid JSON in the format specified by:
                {scoring_parser.get_format_instructions()}"""
        
        #invoke agent
        raw_response = justice_agent.invoke(
                {"messages": [{"role": "system", "content": prompt}]}
            )

        messages = raw_response["messages"]
        response = messages[-1].content

        structured_response = parse_output(state, scoring_retry_parser, response)

        emotions_index = structured_response.find("emotions_scores=")
        motives_index = structured_response.find("motives_scores=")
        knowledge_index = structured_response.find("knowledge_scores=")

        scores = {}
        scores['beliefs'] = structured_response[16:(emotions_index-2)]
        scores['emotions'] = structured_response[(emotions_index+17):(motives_index-2)]
        scores['motives'] = structured_response[(motives_index+16):(knowledge_index-2)]
        scores['knowledge'] = structured_response[(knowledge_index+18):-1]

        print("(4) justice_agent scored hypotheses")
        print("----------------------------")
        print("beliefs:", scores['beliefs'])
        print("emotions:", scores['emotions'])
        print("motives:", scores['motives'])
        print("knowledge:", scores['knowledge'])
        print("----------------------------")

        return {"justice_agent_beliefs_votes": scores['beliefs'],
                "justice_agent_emotions_votes": scores['emotions'],
                "justice_agent_motives_votes": scores['motives'],
                "justice_agent_knowledge_votes": scores['knowledge']}

def utilitarian_agent_debater(state: State):
    if state['debate_step'] == 1: #generate the complete ToM hypothesis
        print("(1) utilitarian_agent generating hypothesis...")

        prompt = f"""[Context]
                     You are presented with the following query consisting of an ethical question, dilemma, or consideration: "{state['user_prompt']}"

                     [Task]
                     You must generate a theory-of-mind (ToM) hypothesis about the entity most relevant to the query. The identity of the entity, or "subject" of the hypothesis, depends on the nature of the query. It could be the person who wrote the query, an entity mentioned in the query, the general human stakeholder, society at large, or any other real or hypothetical person or organization whose internal deliberative process should influence a response to the query.

                     Address the following four sub-hypotheses in your response:

                     1. "beliefs": What beliefs could the subject hold that may have influenced the query? What cultural assumptions or ideological premises might the subject maintain about the issue, value, or any other variable in question, especially as they relate to ethics?
                     2. "emotions": What affective dependencies or triggers may have influenced the way the subject processes the variable in question? 
                     3. "motives": What does the subject aim to achieve by resolving the query? How is the query relevant to the subject's goals?
                     4. "knowledge": What certain facts or reality is the subject aware of in the context of the query? How does that knowledge impact the way the subject processes the query?

                     Be sure to clearly identify the subject's identity in your response. Limit each
                     sub-hypothesis to 50 or fewer words. Adhere to the JSON format specified by
                     {hypothesis_parser.get_format_instructions()}"""
        
        #invoke agent
        raw_response = utilitarian_agent.invoke(
                {"messages": [{"role": "system", "content": prompt}]}
            )

        messages = raw_response["messages"]
        response = messages[-1].content

        structured_response = parse_output(state, hypothesis_retry_parser, response)

        print("(1) utilitarian_agent generated hypothesis")
        print("-------------------------------")
        print(structured_response)
        print("-------------------------------")
        
        #store response as dictionary
        hypothesis = {}
        emotions_index = structured_response.find("emotions=")
        motives_index = structured_response.find("motives=")
        knowledge_index = structured_response.find("knowledge=")

        #cut out stuff like "knowledge=" and extraneous apostrophes
        hypothesis['beliefs'] = structured_response[9:(emotions_index-2)]
        hypothesis['emotions'] = structured_response[(emotions_index+10):(motives_index-2)]
        hypothesis['motives'] = structured_response[(motives_index+9):(knowledge_index-2)]
        hypothesis['knowledge'] = structured_response[(knowledge_index+11):-1]

        return {'utilitarian_beliefs': hypothesis['beliefs'],
                'utilitarian_emotions': hypothesis['emotions'],
                'utilitarian_motives': hypothesis['motives'],
                'utilitarian_knowledge': hypothesis['knowledge']}

    elif state['debate_step'] == 2: #provide feedback *based on most disagreed component*
        print("(2) utilitarian_agent generating feedback...")
        
        #invoke agent
        raw_response = utilitarian_agent.invoke(
                {"messages": [{"role": "system", "content": f"""
                                User prompt: {state['user_prompt']}.
                                Hypotheses generated by the justice, utilitarian,
                                and common-good agents, respectively:
                                Hypothesis (a): {state[f'justice_{state['worst_component']}']}
                                Hypothesis (b): {state[f'utilitarian_{state['worst_component']}']}
                                Hypothesis (c): {state[f'common_good_{state['worst_component']}']}

                                Evaluate hypotheses (a), (b), and (c) on the following
                                criteria:
                                1. Evidential support: Does the information in the user prompt
                                directly support each claim made about the subject of the
                                user prompt's inner
                                state, in relation to the subject's {state['worst_component']}?
                                Identify the least supported claim or claims and briefly
                                explain how they should be revised or improved.
                                2. Belief projection: Does the hypothesis wrongly attribute
                                the thought process of an ethical reasoning agent to the subject of
                                the user prompt? Specifically, do the ethics agents that
                                produced the hypothesis (care, justice, utilitarian, and
                                common-good ethics agents) inappropriately project their own
                                ethical worldviews about the subject's {state['worst_component']}
                                onto the subject? Identify any example(s) of belief projection
                                in a given hypothesis and briefly explain how they should be
                                revised or improved.
                                3. Plausibility: Given the user prompt and what it suggests
                                about the subject, could the claims of each hypothesis
                                plausibly reflect the subject's inner state of {state['worst_component']}?
                                Identify the most implausible component(s) of each hypothesis
                                and briefly explain how they should be revised or improved.
                                4. General improvements: How else could the hypothesis
                                improve?

                                Your response will include each of the four evaluation
                                components listed above for each hypothesis. Keep each
                                evaluation component very brief and to-the-point; it's
                                okay if you can't address every single issue at once.
                                Each evaluation component must be 15 or fewer words in
                                length.  Maintain JSON format specified by
                                {feedback_parser.get_format_instructions()}
                                """}]}
            )

        messages = raw_response["messages"]
        response = messages[-1].content

        structured_response = parse_output(state, feedback_retry_parser, response)

        a2_index = structured_response.find("hypothesis_a_evidence=")
        a3_index = structured_response.find("hypothesis_a_projection=")
        a4_index = structured_response.find("hypothesis_a_other=")

        b1_index = structured_response.find("hypothesis_b_plausibility=")
        b2_index = structured_response.find("hypothesis_b_evidence=")
        b3_index = structured_response.find("hypothesis_b_projection=")
        b4_index = structured_response.find("hypothesis_b_other=")

        c1_index = structured_response.find("hypothesis_c_plausibility=")
        c2_index = structured_response.find("hypothesis_c_evidence=")
        c3_index = structured_response.find("hypothesis_c_projection=")
        c4_index = structured_response.find("hypothesis_c_other=")

        a1 = structured_response[27:(a2_index-2)]
        a2 = structured_response[(a2_index+23):(a3_index-2)]
        a3 = structured_response[(a3_index+25):(a4_index-2)]
        a4 = structured_response[(a4_index+20):(b1_index-2)]
        
        b1 = structured_response[(b1_index+27):(b2_index-2)]
        b2 = structured_response[(b2_index+23):(b3_index-2)]
        b3 = structured_response[(b3_index+25):(b4_index-2)]
        b4 = structured_response[(b4_index+20):(c1_index-2)]

        c1 = structured_response[(c1_index+27):(c2_index-2)]
        c2 = structured_response[(c2_index+23):(c3_index-2)]
        c3 = structured_response[(c3_index+25):(c4_index-2)]
        c4 = structured_response[(c4_index+20):-1]

        feedback = {}
        feedback['hypothesis_a'] = a1 + " " + a2 + " " + a3 + " " + a4
        feedback['hypothesis_b'] = b1 + " " + b2 + " " + b3 + " " + b4
        feedback['hypothesis_c'] = c1 + " " + c2 + " " + c3 + " " + c4

        print("(2) utilitarian_agent generated feedback")
        print("-------------------------------")
        print("care feedback:", feedback['hypothesis_a'])
        print("justice feedback:", feedback['hypothesis_b'])
        print("common good feedback:", feedback['hypothesis_c'])
        print("-------------------------------")

        return {"care_agent_feedback": [feedback['hypothesis_a']],
                "justice_agent_feedback": [feedback['hypothesis_b']],
                "common_good_agent_feedback": [feedback['hypothesis_c']]}

    elif state['debate_step'] == 3: #revise hypothesis
        print(f"(3) utilitarian_agent re-generating {state['worst_component']} hypothesis...")

        #create feedback string from relevant parts of feedback array
        feedback = ""
        for i in range(3):
            s = state['utilitarian_agent_feedback'][i + (state['rounds_performed'] * 3)]
            feedback += s + " "

        generation_instructions = "" #specific instructions for generating the
                                        #most disagreed sub-component

        match state['worst_component']:
            case "beliefs":
                generation_instructions += """Summarize, in under 50 words, what you think the subject
                                              of the user prompt believes about the situation described
                                              in the prompt."""
            case "emotions":
                generation_instructions += """Summarize, in under 50 words, how you think the subject
                                              of the user prompt feels about the situation described
                                              in their prompt. Be specific."""
            case "motives":
                generation_instructions += """Summarize, in under 50 words, the motives and goals you think
                                              the subject of the user prompt possesses in relation to the
                                              situation described in the prompt."""
                
            case "knowledge":
                generation_instructions += """Summarize, in under 50 words, what you think the subject
                                              of the user prompt knows for sure about the situation they
                                              described in the prompt. Be specific."""

        #invoke agent
        raw_response = utilitarian_agent.invoke(
                {"messages": [{"role": "system", "content": f"""
                                User prompt: {state['user_prompt']}. Your original
                                hypothesis on the {state['worst_component']} of the
                                subject of the user prompt:
                                {state[f'utilitarian_{state['worst_component']}']}.
                                Feedback from other agents on how to improve
                                your original hypothesis: {feedback}

                                Based on the user prompt, the original hypothesis,
                                and the feedback on how to improve your original
                                hypothesis, generate an improved hypothesis about the
                                {state['worst_component']} of the subject of the user
                                prompt. Be open to the other agents' unique moral
                                perspectives. {generation_instructions} Additionally,
                                your improved hypothesis should be plausible
                                and supported by information in the user prompt. Be
                                specific. Limit your response to fewer than 50 words in length.

                                Adhere to the JSON format specified by
                                {response_parser.get_format_instructions()}
                                """}]}
            )

        messages = raw_response["messages"]
        response = messages[-1].content

        if "response=" in response: #potential for empty string due to total parsing failure!
            structured_response = parse_output(state, response_retry_parser, response).split("response=")[1] #try to extract only the actual response
        else:
            structured_response = parse_output(state, response_retry_parser, response)

        print(f"(3) utilitarian_agent re-generated {state['worst_component']} hypothesis")
        print("-------------------------------")
        print(structured_response)
        print("-------------------------------")

        return {f'utilitarian_{state['worst_component']}': structured_response}

    elif state['debate_step'] == 4: #"score" other responses (ie, vote)
        print("(4) utilitarian_agent scoring hypotheses...")

        prompt = f"""
                User prompt: {state['user_prompt']}.
                Hypotheses about the beliefs, emotions, motives,
                and knowledge of the subject of the user prompt:
                Beliefs hypothesis by care agent: {state['care_beliefs']}
                Beliefs hypothesis by justice agent: {state['justice_beliefs']}
                Beliefs hypothesis by common_good agent: {state['common_good_beliefs']}

                Emotions hypothesis by care agent: {state['care_emotions']}
                Emotions hypothesis by justice agent: {state['justice_emotions']}
                Emotions hypothesis by common_good agent: {state['common_good_emotions']}

                Motives hypothesis by care agent: {state['care_motives']}
                Motives hypothesis by justice agent: {state['justice_motives']}
                Motives hypothesis by common_good agent: {state['common_good_motives']}

                Knowledge hypothesis by care agent: {state['care_knowledge']}
                Knowledge hypothesis by justice agent: {state['justice_knowledge']}
                Knowledge hypothesis by common_good agent: {state['common_good_knowledge']}

                For each individual hypothesis in every group -- that is,
                for each beliefs hypothesis, emotions hypothesis, motives
                hypothesis, and knowledge hypothesis on its own -- think
                about how well it theorizes about the state of mind of the
                subject of the user prompt. Conisder the following key
                questions as you think: What is implausible about each
                interpretation? Is each hypothesis supported by evidence
                from the user prompt? Does each hypothesis wrongly assign
                an ethical reasoning agent's beliefs to the subject, even if the
                subject isn't knowledgeable in ethics?

                Give each hypothesis a score in [1, 10]. Be
                critical; it's okay to give a low score. Scores *must* be a
                single number. Your response will be organized by category.
                For each hypothesis category, return three scores (one
                for each hypothesis in the category) in the following
                format: [score (care agent)] | [score (justice agent)] | [score (common_good agent)] (replacing
                the bracketed statements with your score for that hypothesis
                in a given category). For example, a correct output for
                the hypothetical "intentions" category could look like
                "3 | 7 | 4".

                Return ONLY valid JSON in the format specified by:
                {scoring_parser.get_format_instructions()}"""
        
        #invoke agent
        raw_response = utilitarian_agent.invoke(
                {"messages": [{"role": "system", "content": prompt}]}
            )

        messages = raw_response["messages"]
        response = messages[-1].content

        structured_response = parse_output(state, scoring_retry_parser, response)

        emotions_index = structured_response.find("emotions_scores=")
        motives_index = structured_response.find("motives_scores=")
        knowledge_index = structured_response.find("knowledge_scores=")

        scores = {}
        scores['beliefs'] = structured_response[16:(emotions_index-2)]
        scores['emotions'] = structured_response[(emotions_index+17):(motives_index-2)]
        scores['motives'] = structured_response[(motives_index+16):(knowledge_index-2)]
        scores['knowledge'] = structured_response[(knowledge_index+18):-1]

        print("(4) utilitarian_agent scored hypotheses")
        print("----------------------------")
        print("beliefs:", scores['beliefs'])
        print("emotions:", scores['emotions'])
        print("motives:", scores['motives'])
        print("knowledge:", scores['knowledge'])
        print("----------------------------")

        return {"utilitarian_agent_beliefs_votes": scores['beliefs'],
                "utilitarian_agent_emotions_votes": scores['emotions'],
                "utilitarian_agent_motives_votes": scores['motives'],
                "utilitarian_agent_knowledge_votes": scores['knowledge']}

def common_good_agent_debater(state: State):
    if state['debate_step'] == 1: #generate the complete ToM hypothesis
        print("(1) common_good_agent generating hypothesis...")

        prompt = f"""[Context]
                     You are presented with the following query consisting of an ethical question, dilemma, or consideration: "{state['user_prompt']}"

                     [Task]
                     You must generate a theory-of-mind (ToM) hypothesis about the entity most relevant to the query. The identity of the entity, or "subject" of the hypothesis, depends on the nature of the query. It could be the person who wrote the query, an entity mentioned in the query, the general human stakeholder, society at large, or any other real or hypothetical person or organization whose internal deliberative process should influence a response to the query.

                     Address the following four sub-hypotheses in your response:

                     1. "beliefs": What beliefs could the subject hold that may have influenced the query? What cultural assumptions or ideological premises might the subject maintain about the issue, value, or any other variable in question, especially as they relate to ethics?
                     2. "emotions": What affective dependencies or triggers may have influenced the way the subject processes the variable in question? 
                     3. "motives": What does the subject aim to achieve by resolving the query? How is the query relevant to the subject's goals?
                     4. "knowledge": What certain facts or reality is the subject aware of in the context of the query? How does that knowledge impact the way the subject processes the query?

                    You must *explicitly* identify the subject's identity in your response (for example,
                    "the user believes that..." or "society assumes the value at hand implies...").
                    Limit each sub-hypothesis to 50 or fewer words. Adhere to the JSON format specified by
                    {hypothesis_parser.get_format_instructions()}"""
        
        #invoke agent
        raw_response = common_good_agent.invoke(
                {"messages": [{"role": "system", "content": prompt}]}
            )

        messages = raw_response["messages"]
        response = messages[-1].content

        structured_response = parse_output(state, hypothesis_retry_parser, response)

        print("(1) common_good_agent generated hypothesis")
        print("-------------------------------")
        print(structured_response)
        print("-------------------------------")
        
        #store response as dictionary
        hypothesis = {}
        emotions_index = structured_response.find("emotions=")
        motives_index = structured_response.find("motives=")
        knowledge_index = structured_response.find("knowledge=")

        #cut out stuff like "knowledge=" and extraneous apostrophes
        hypothesis['beliefs'] = structured_response[9:(emotions_index-2)]
        hypothesis['emotions'] = structured_response[(emotions_index+10):(motives_index-2)]
        hypothesis['motives'] = structured_response[(motives_index+9):(knowledge_index-2)]
        hypothesis['knowledge'] = structured_response[(knowledge_index+11):-1]

        return {'common_good_beliefs': hypothesis['beliefs'],
                'common_good_emotions': hypothesis['emotions'],
                'common_good_motives': hypothesis['motives'],
                'common_good_knowledge': hypothesis['knowledge']}

    elif state['debate_step'] == 2: #provide feedback *based on most disagreed component*
        print("(2) common_good_agent generating feedback...")
        
        #invoke agent
        raw_response = common_good_agent.invoke(
                {"messages": [{"role": "system", "content": f"""
                                User prompt: {state['user_prompt']}.
                                Hypotheses generated by the justice, utilitarian,
                                and common-good agents, respectively:
                                Hypothesis (a): {state[f'justice_{state['worst_component']}']}
                                Hypothesis (b): {state[f'utilitarian_{state['worst_component']}']}
                                Hypothesis (c): {state[f'common_good_{state['worst_component']}']}

                                Evaluate hypotheses (a), (b), and (c) on the following
                                criteria:
                                1. Evidential support: Does the information in the user prompt
                                directly support each claim made about the subject of the
                                user prompt's inner
                                state, in relation to the subject's {state['worst_component']}?
                                Identify the least supported claim or claims and briefly
                                explain how they should be revised or improved.
                                2. Belief projection: Does the hypothesis wrongly attribute
                                the thought process of an ethical reasoning agent to the subject of
                                the user prompt? Specifically, do the ethics agents that
                                produced the hypothesis (care, justice, utilitarian, and
                                common-good ethics agents) inappropriately project their own
                                ethical worldviews about the subject's {state['worst_component']}
                                onto the subject? Identify any example(s) of belief projection
                                in a given hypothesis and briefly explain how they should be
                                revised or improved.
                                3. Plausibility: Given the user prompt and what it suggests
                                about the subject, could the claims of each hypothesis
                                plausibly reflect the subject's inner state of {state['worst_component']}?
                                Identify the most implausible component(s) of each hypothesis
                                and briefly explain how they should be revised or improved.
                                4. General improvements: How else could the hypothesis
                                improve?

                                Your response will include each of the four evaluation
                                components listed above for each hypothesis. Keep each
                                evaluation component very brief and to-the-point; it's
                                okay if you can't address every single issue at once.
                                Each evaluation component must be 15 or fewer words in
                                length.  Maintain JSON format specified by
                                {feedback_parser.get_format_instructions()}
                                """}]}
            )

        messages = raw_response["messages"]
        response = messages[-1].content

        structured_response = parse_output(state, feedback_retry_parser, response)

        a2_index = structured_response.find("hypothesis_a_evidence=")
        a3_index = structured_response.find("hypothesis_a_projection=")
        a4_index = structured_response.find("hypothesis_a_other=")

        b1_index = structured_response.find("hypothesis_b_plausibility=")
        b2_index = structured_response.find("hypothesis_b_evidence=")
        b3_index = structured_response.find("hypothesis_b_projection=")
        b4_index = structured_response.find("hypothesis_b_other=")

        c1_index = structured_response.find("hypothesis_c_plausibility=")
        c2_index = structured_response.find("hypothesis_c_evidence=")
        c3_index = structured_response.find("hypothesis_c_projection=")
        c4_index = structured_response.find("hypothesis_c_other=")

        a1 = structured_response[27:(a2_index-2)]
        a2 = structured_response[(a2_index+23):(a3_index-2)]
        a3 = structured_response[(a3_index+25):(a4_index-2)]
        a4 = structured_response[(a4_index+20):(b1_index-2)]
        
        b1 = structured_response[(b1_index+27):(b2_index-2)]
        b2 = structured_response[(b2_index+23):(b3_index-2)]
        b3 = structured_response[(b3_index+25):(b4_index-2)]
        b4 = structured_response[(b4_index+20):(c1_index-2)]

        c1 = structured_response[(c1_index+27):(c2_index-2)]
        c2 = structured_response[(c2_index+23):(c3_index-2)]
        c3 = structured_response[(c3_index+25):(c4_index-2)]
        c4 = structured_response[(c4_index+20):-1]

        feedback = {}
        feedback['hypothesis_a'] = a1 + " " + a2 + " " + a3 + " " + a4
        feedback['hypothesis_b'] = b1 + " " + b2 + " " + b3 + " " + b4
        feedback['hypothesis_c'] = c1 + " " + c2 + " " + c3 + " " + c4

        print("(2) common_good_agent generated feedback")
        print("-------------------------------")
        print("care feedback:", feedback['hypothesis_a'])
        print("justice feedback:", feedback['hypothesis_b'])
        print("utilitarian feedback:", feedback['hypothesis_c'])
        print("-------------------------------")

        return {"care_agent_feedback": [feedback['hypothesis_a']],
                "justice_agent_feedback": [feedback['hypothesis_b']],
                "utilitarian_agent_feedback": [feedback['hypothesis_c']]}

    elif state['debate_step'] == 3: #revise hypothesis
        print(f"(3) common_good_agent re-generating {state['worst_component']} hypothesis...")

        #create feedback string from relevant parts of feedback array
        feedback = ""
        for i in range(3):
            s = state['common_good_agent_feedback'][i + (state['rounds_performed'] * 3)]
            feedback += s + " "

        generation_instructions = "" #specific instructions for generating the
                                        #most disagreed sub-component

        match state['worst_component']:
            case "beliefs":
                generation_instructions += """Summarize, in under 50 words, what you think the subject
                                              of the user prompt believes about the situation described
                                              in the prompt."""
            case "emotions":
                generation_instructions += """Summarize, in under 50 words, how you think the subject
                                              of the user prompt feels about the situation described
                                              in their prompt. Be specific."""
            case "motives":
                generation_instructions += """Summarize, in under 50 words, the motives and goals you think
                                              the subject of the user prompt possesses in relation to the
                                              situation described in the prompt."""
                
            case "knowledge":
                generation_instructions += """Summarize, in under 50 words, what you think the subject
                                              of the user prompt knows for sure about the situation they
                                              described in the prompt. Be specific."""

        #invoke agent
        raw_response = common_good_agent.invoke(
                {"messages": [{"role": "system", "content": f"""
                                User prompt: {state['user_prompt']}. Your original
                                hypothesis on the {state['worst_component']} of the
                                subject of the user prompt:
                                {state[f'common_good_{state['worst_component']}']}.
                                Feedback from other agents on how to improve
                                your original hypothesis: {feedback}

                                Based on the user prompt, the original hypothesis,
                                and the feedback on how to improve your original
                                hypothesis, generate an improved hypothesis about the
                                {state['worst_component']} of the subject of the user
                                prompt. Be open to the other agents' unique moral
                                perspectives. {generation_instructions} Additionally,
                                your improved hypothesis should be plausible
                                and supported by information in the user prompt. Be
                                specific. Limit your response to fewer than 50 words in length.

                                Adhere to the JSON format specified by
                                {response_parser.get_format_instructions()}
                                """}]}
            )

        messages = raw_response["messages"]
        response = messages[-1].content

        if "response=" in response: #potential for empty string due to total parsing failure!
            structured_response = parse_output(state, response_retry_parser, response).split("response=")[1] #try to extract only the actual response
        else:
            structured_response = parse_output(state, response_retry_parser, response)

        print(f"(3) common_good_agent re-generated {state['worst_component']} hypothesis")
        print("-------------------------------")
        print(structured_response)
        print("-------------------------------")

        return {f'common_good_{state['worst_component']}': structured_response}

    elif state['debate_step'] == 4: #"score" other responses (ie, vote)
        print("(4) common_good_agent scoring hypotheses...")

        prompt = f"""
                User prompt: {state['user_prompt']}.
                Hypotheses about the beliefs, emotions, motives,
                and knowledge of the subject of the user prompt:
                Beliefs hypothesis by care agent: {state['care_beliefs']}
                Beliefs hypothesis by justice agent: {state['justice_beliefs']}
                Beliefs hypothesis by utilitarian agent: {state['utilitarian_beliefs']}

                Emotions hypothesis by care agent: {state['care_emotions']}
                Emotions hypothesis by justice agent: {state['justice_emotions']}
                Emotions hypothesis by utilitarian agent: {state['utilitarian_emotions']}

                Motives hypothesis by care agent: {state['care_motives']}
                Motives hypothesis by justice agent: {state['justice_motives']}
                Motives hypothesis by utilitarian agent: {state['utilitarian_motives']}

                Knowledge hypothesis by care agent: {state['care_knowledge']}
                Knowledge hypothesis by justice agent: {state['justice_knowledge']}
                Knowledge hypothesis by utilitarian agent: {state['utilitarian_knowledge']}

                For each individual hypothesis in every group -- that is,
                for each beliefs hypothesis, emotions hypothesis, motives
                hypothesis, and knowledge hypothesis on its own -- think
                about how well it theorizes about the state of mind of the
                subject of the user prompt. Conisder the following key
                questions as you think: What is implausible about each
                interpretation? Is each hypothesis supported by evidence
                from the user prompt? Does each hypothesis wrongly assign
                an ethical reasoning agent's beliefs to the subject, even if the
                subject isn't knowledgeable in ethics?

                Give each hypothesis a score in [1, 10]. Be
                critical; it's okay to give a low score. Scores *must* be a
                single number. Your response will be organized by category.
                For each hypothesis category, return three scores (one
                for each hypothesis in the category) in the following
                format: [score (care agent)] | [score (justice agent)] | [score (utilitarian agent)] (replacing
                the bracketed statements with your score for that hypothesis
                in a given category). For example, a correct output for
                the hypothetical "intentions" category could look like
                "3 | 7 | 4".

                Return ONLY valid JSON in the format specified by:
                {scoring_parser.get_format_instructions()}"""
        
        #invoke agent
        raw_response = common_good_agent.invoke(
                {"messages": [{"role": "system", "content": prompt}]}
            )

        messages = raw_response["messages"]
        response = messages[-1].content

        structured_response = parse_output(state, scoring_retry_parser, response)

        emotions_index = structured_response.find("emotions_scores=")
        motives_index = structured_response.find("motives_scores=")
        knowledge_index = structured_response.find("knowledge_scores=")

        scores = {}
        scores['beliefs'] = structured_response[16:(emotions_index-2)]
        scores['emotions'] = structured_response[(emotions_index+17):(motives_index-2)]
        scores['motives'] = structured_response[(motives_index+16):(knowledge_index-2)]
        scores['knowledge'] = structured_response[(knowledge_index+18):-1]

        print("(4) common_good_agent scored hypotheses")
        print("----------------------------")
        print("beliefs:", scores['beliefs'])
        print("emotions:", scores['emotions'])
        print("motives:", scores['motives'])
        print("knowledge:", scores['knowledge'])
        print("----------------------------")

        return {"common_good_agent_beliefs_votes": scores['beliefs'],
                "common_good_agent_emotions_votes": scores['emotions'],
                "common_good_agent_motives_votes": scores['motives'],
                "common_good_agent_knowledge_votes": scores['knowledge']}

def moderator(state: State):
    """ manages debate stages """

    if state['debate_step'] == 1:
        print()
        print("(*) starting feedback generation...")
        print()

        return {"debate_step": 2}

    elif state['debate_step'] == 2:
        print()
        print("(*) starting component re-generation...")
        print()

        return {"debate_step": 3}

    elif state['debate_step'] == 3:
        print()
        print("(*) starting debate orchestrator...")
        print()

        return {"debate_step": 3.5,
                "rounds_performed": state['rounds_performed'] + 1}
    
    else: #debate_stage 4
        print()
        print("(*) debate concluded; accumulating votes...")
        print()

        beliefs_evaluations = [state['care_agent_beliefs_votes'],
                               state['justice_agent_beliefs_votes'],
                               state['utilitarian_agent_beliefs_votes'],
                               state['common_good_agent_beliefs_votes']]
        beliefs_hypothesis = get_best_hypothesis(state, beliefs_evaluations, "beliefs")

        motives_evaluations = [state['care_agent_motives_votes'],
                               state['justice_agent_motives_votes'],
                               state['utilitarian_agent_motives_votes'],
                               state['common_good_agent_motives_votes']]
        motives_hypothesis = get_best_hypothesis(state, motives_evaluations, "motives")

        emotions_evaluations = [state['care_agent_emotions_votes'],
                                state['justice_agent_emotions_votes'],
                                state['utilitarian_agent_emotions_votes'],
                                state['common_good_agent_emotions_votes']]
        emotions_hypothesis = get_best_hypothesis(state, emotions_evaluations, "emotions")
    
        knowledge_evaluations = [state['care_agent_knowledge_votes'],
                                 state['justice_agent_knowledge_votes'],
                                 state['utilitarian_agent_knowledge_votes'],
                                 state['common_good_agent_knowledge_votes']]
        knowledge_hypothesis = get_best_hypothesis(state, knowledge_evaluations, "knowledge")

        full_hypothesis = beliefs_hypothesis + " " + motives_hypothesis + " " + \
                          emotions_hypothesis + " " + knowledge_hypothesis

        return {"tom_hypothesis": full_hypothesis}

def _minmax_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Scale a dict of aspect->score into [0,1]. Flat input -> all zeros."""
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo <= 1e-12:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

def compute_disagreement_old_unused(state: "State") -> Dict[str, float]:  # noqa: F821
    """
    Combined per-aspect disagreement (embedding + epistemic), for ALL aspects.
    Returns {aspect: combined_score}. Falls back to embedding-only if logprobs
    are unavailable. This is the single source of truth used both for selection
    and for history/convergence tracking.
    """
    # (1) embedding disagreement
    sim_report = full_similarity_report(state)  # noqa: F821
    embed_dis = {
        aspect: float(sim_report["aspect_avg_disagreement"].get(aspect, 0.0))
        for aspect in ASPECTS  # noqa: F821
    }
 
    # (2) epistemic disagreement (logprobs)
    epistemic = {aspect: 0.0 for aspect in ASPECTS}  # noqa: F821
    logprobs_ok = True
    try:
        for aspect in ASPECTS:  # noqa: F821
            candidates_by_agent = {
                agent: state.get(f"{agent}_{aspect}", "") for agent in AGENT_ORDER  # noqa: F821
            }
            rep = aspect_logprob_report(  # noqa: F821
                aspect=aspect,
                schemas=AGENT_SCHEMAS,        # noqa: F821
                lens_names=AGENT_LENS_NAMES,  # noqa: F821
                candidates_by_agent=candidates_by_agent,
                user_prompt=state["user_prompt"],
            )
            epistemic[aspect] = float(rep["epistemic_disagreement"])
    except LogprobsUnavailable as e:  # noqa: F821
        logprobs_ok = False
        print(f"** orchestrator: logprobs unavailable ({e}); embedding-only **")
    except Exception as e:
        logprobs_ok = False
        print(f"** orchestrator: logprob scoring failed ({e}); embedding-only **")
 
    # combine
    if not logprobs_ok:
        combined = dict(embed_dis)
    else:
        e_norm = _minmax_normalize(embed_dis)
        p_norm = _minmax_normalize(epistemic)
        combined = {
            aspect: ALPHA * e_norm[aspect] + (1.0 - ALPHA) * p_norm[aspect]
            for aspect in ASPECTS  # noqa: F821
        }
 
    print("** per-aspect disagreement **")
    for aspect in ASPECTS:  # noqa: F821
        print(f"   {aspect:9s}  embed={embed_dis[aspect]:.3f}  "
              f"epistemic={epistemic[aspect]:.3f}  combined={combined[aspect]:.3f}")
    return combined

def compute_disagreement(state: "State") -> Dict[str, float]:
    """Now returns per-component NEED-TO-DEBATE (embedding pre-filter + moral
    rubric). Higher = worse quality = more worth debating. Logprob path retired."""
    resolved = state.get("resolved_aspects", [])
    return compute_need_to_debate(
        state,
        full_similarity_report_fn=full_similarity_report,
        framework_agent=framework_agent,
        sub_score_parser=sub_score_parser,
        sub_score_retry_parser=sub_score_retry_parser,
        parse_output_fn=parse_output,
        aspects=ASPECTS,
        agents=AGENT_ORDER,
        resolved=resolved,
        k=2,
    )

def _pick_new_aspect(
    combined: Dict[str, float],
    resolved: List[str],
) -> str:
    """Highest-disagreement aspect that hasn't been resolved/retired yet."""
    candidates = {a: s for a, s in combined.items() if a not in resolved}
    if not candidates:
        return ""
    return max(candidates, key=candidates.get)
 
##debate parameters (tunable)
# Blend of embedding vs epistemic disagreement. 0=logprob only, 1=embedding only, 0.5 == equal for both
ALPHA = 0.5
 
# An aspect counts as RESOLVED when its combined disagreement drops below this.
CONVERGENCE_THRESHOLD = 0.15
 
# An aspect counts as PLATEAUED when this round's disagreement dropped by less
# than this vs the previous round (stopped improving -> stop re-debating it).
MIN_IMPROVEMENT = 0.03

def debate_orchestrator(state: "State"):  # noqa: F821
    """
    Two entry contexts (unchanged control flow via debate_step):
 
    * debate_step == 1  -> just came from initial full-hypothesis generation.
                           Lock the most-disagreed aspect and start the debate.
    * otherwise (3.5)   -> came back from the moderator after a full
                           feedback->revise->score round on the locked aspect.
                           Record history, then make the 3-way decision.
    """
    # Hard backstop: global round budget.
    if state["rounds_performed"] >= state["max_rounds"]:
        print("** orchestrator: round limit reached -> voting **")
        return {"debate_step": 4}
 
    combined = compute_disagreement(state)
 
    # ---- first entry: choose and lock an aspect, begin debating --------------
    if state["debate_step"] == 1:
        resolved = state.get("resolved_aspects", [])
        locked = _pick_new_aspect(combined, resolved)
        history = state.get("disagreement_history", [])
        history = history + [combined]  # snapshot the starting point
        print(f"** orchestrator locked aspect: {locked} **")
        return {
            "worst_component": locked,
            "locked_aspect": locked,
            "disagreement_history": history,
        }
 
    # ---- return from moderator: record history + 3-way decision -------------
    locked = state.get("locked_aspect") or state["worst_component"]
    resolved = list(state.get("resolved_aspects", []))
    history = list(state.get("disagreement_history", []))
 
    curr = combined.get(locked, 1.0)
    prev = history[-1].get(locked, 1.0) if history else 1.0
    drop = prev - curr
    history.append(combined)  # append this round's full snapshot
 
    print(f"** aspect '{locked}': prev={prev:.3f} -> curr={curr:.3f} "
          f"(drop {drop:+.3f}) **")
 
    resolved_now = curr < CONVERGENCE_THRESHOLD
    plateaued = (drop < MIN_IMPROVEMENT) and not resolved_now
 
    if resolved_now or plateaued:
        # retire this aspect so we never re-debate it
        if locked not in resolved:
            resolved.append(locked)
        reason = "RESOLVED" if resolved_now else "PLATEAUED (irreducible)"
        print(f"** aspect '{locked}' -> {reason}; retiring it **")
 
        next_aspect = _pick_new_aspect(combined, resolved)
        if not next_aspect:
            print("** no aspects left to debate -> voting **")
            return {
                "debate_step": 4,
                "resolved_aspects": resolved,
                "disagreement_history": history,
            }
 
        print(f"** switching debate to new aspect: {next_aspect} **")
        return {
            "worst_component": next_aspect,
            "locked_aspect": next_aspect,
            "resolved_aspects": resolved,
            "disagreement_history": history,
            "debate_step": 1,   # continue debating (feedback->revise) on new aspect
        }
 
    # still improving and not converged -> another round on the SAME aspect
    print(f"** aspect '{locked}' still improving -> another round **")
    return {
        "worst_component": locked,
        "locked_aspect": locked,
        "resolved_aspects": resolved,
        "disagreement_history": history,
        "debate_step": 1,
    }

#control flow nodes
def return_to_debate(state: State):
    return

#conditional checks
def check_step_1(state: State): #to orchestrator from agents, or to moderator?
    """ returns:
            PASS if going to orchestrator from agents, or
            FAIL if going to moderator from agents """

    if state['debate_step'] == 1:
        return "PASS"
    else:
        return "FAIL"

def check_step_4(state: State): #to moderator from orchestrator, or to agents for voting?
    """ returns:
            PASS if going to agents from orchestrator, or
            FAIL if going to moderator from orchestrator """

    if state['debate_step'] == 4:
        return "PASS"
    else:
        return "FAIL"

def check_moderator_stage(state: State): #coming from moderator
    """ returns:
            ORCH if moving on to orchestrator,
            PASS if debate over, or
            FAIL otherwise """

    if state['debate_step'] == 3.5:
        return "ORCH"
    elif state['debate_step'] == 4:
        return "PASS"
    else:
        return "FAIL"

#helper functions
def parse_output(state: State, parser, response) -> str:
    try:
        structured_response = parser.parse(
            completion = response
        )

        #log whatever the agent just did
        log.agent_step(
            event="Agent generated final response.",
            action="Final Answer",
            action_input=f"{state['user_prompt']}",
            output=response
        )
    except Exception as e:
        print("\nError parsing response:\n")
        print(e)

        print("\nRaw Response:\n")
        print(response)

        #log the agent's unfortunate & embarrassing failure
        log.agent_step(
            event="Parser failed.",
            action="Parse Error",
            action_input=f"{state['user_prompt']}",
            output=response
        )

        #dummy response in case of parsing failure
        structured_response = ""
    
    return str(structured_response)

def get_best_hypothesis(state: State, evaluations, component) -> str:
    care_agent_total_votes = 0
    justice_agent_total_votes = 0
    utilitarian_agent_total_votes = 0
    common_good_agent_total_votes = 0

    for i in range(len(evaluations)):
        response = evaluations[i]
        votes = re.findall(r'-?\d*\.?\d+', response)

        match i:
            case 0: #care agent
                if len(votes) >= 3: #guard against bad output
                    justice_agent_total_votes += float(votes[0])
                    utilitarian_agent_total_votes += float(votes[1])
                    common_good_agent_total_votes += float(votes[2])
            case 1: #justice agent
                if len(votes) >= 3:
                    care_agent_total_votes += float(votes[0])
                    utilitarian_agent_total_votes += float(votes[1])
                    common_good_agent_total_votes += float(votes[2])
            case 2: #utilitarian agent
                if len(votes) >= 3:
                    care_agent_total_votes += float(votes[0])
                    justice_agent_total_votes += float(votes[1])
                    common_good_agent_total_votes += float(votes[2])
            case 3: #common_good agent
                if len(votes) >= 3:
                    care_agent_total_votes += float(votes[0])
                    justice_agent_total_votes += float(votes[1])
                    utilitarian_agent_total_votes += float(votes[2])

    #make each key unique in case multiple agents get the same score
    hypotheses = {
            str(care_agent_total_votes) + "c": "care_agent",
            str(justice_agent_total_votes) + "j": "justice_agent",
            str(utilitarian_agent_total_votes) + "u": "utilitarian_agent",
            str(common_good_agent_total_votes) + "g": "common_good_agent"
        }

    highest_score = -1
    highest_votes = "" #key from hypotheses dict

    #find highest score value, corresponding key
    for v in hypotheses:
        score = float(re.findall(r'-?\d*\.?\d+', v)[0])
        if score > highest_score:
            highest_score = score
            highest_votes = v
    
    #find potential ties for highest score
    same_highest_votes = [highest_votes] #keys whose scores tie for highest
                                            #(could end up w/length 1!)
    for v in hypotheses:
        if v != same_highest_votes[0]:
            score = float(re.findall(r'-?\d*\.?\d+', v)[0])
            if score == highest_score:
                same_highest_votes.append(v)

    #randomly pick among hypotheses tied for the highest score
    best_hypothesis = hypotheses[random.choice(same_highest_votes)]

    print(f"best {component} hypothesis:", best_hypothesis.upper(), "with",
            highest_score, "total votes and", (len(same_highest_votes) - 1), "ties")

    match best_hypothesis:
        case "care_agent":
            return state[f'care_{component}']
        case "justice_agent":
            return state[f'justice_{component}']
        case "utilitarian_agent":
            return state[f'utilitarian_{component}']
        case "common_good_agent":
            return state[f'common_good_{component}']

## BUILD GRAPH
generate_and_refine_hypothesis = StateGraph(State)

#add nodes
generate_and_refine_hypothesis.add_node("care_agent debater", care_agent_debater)
generate_and_refine_hypothesis.add_node("justice_agent debater", justice_agent_debater)
generate_and_refine_hypothesis.add_node("utilitarian_agent debater", utilitarian_agent_debater)
generate_and_refine_hypothesis.add_node("common_good_agent debater", common_good_agent_debater)

generate_and_refine_hypothesis.add_node("moderator", moderator)

generate_and_refine_hypothesis.add_node("debate_orchestrator", debate_orchestrator)

generate_and_refine_hypothesis.add_node("return to debate", return_to_debate)

#add edges
generate_and_refine_hypothesis.add_edge(START, "care_agent debater")
generate_and_refine_hypothesis.add_edge(START, "justice_agent debater")
generate_and_refine_hypothesis.add_edge(START, "utilitarian_agent debater")
generate_and_refine_hypothesis.add_edge(START, "common_good_agent debater")

#i need to go from the debaters to the orchestrator *only* in stage 1
generate_and_refine_hypothesis.add_conditional_edges(
        "care_agent debater", check_step_1, {"PASS": "debate_orchestrator", "FAIL": "moderator"}
    )
generate_and_refine_hypothesis.add_conditional_edges(
        "justice_agent debater", check_step_1, {"PASS": "debate_orchestrator", "FAIL": "moderator"}
    )
generate_and_refine_hypothesis.add_conditional_edges(
        "utilitarian_agent debater", check_step_1, {"PASS": "debate_orchestrator", "FAIL": "moderator"}
    )
generate_and_refine_hypothesis.add_conditional_edges(
        "common_good_agent debater", check_step_1, {"PASS": "debate_orchestrator", "FAIL": "moderator"}
    )

generate_and_refine_hypothesis.add_conditional_edges(
        "debate_orchestrator", check_step_4, {"PASS": "return to debate", "FAIL": "moderator"}
    )

generate_and_refine_hypothesis.add_conditional_edges(
        "moderator", check_moderator_stage, {"ORCH": "debate_orchestrator", "PASS": END, "FAIL": "return to debate"}
    )

generate_and_refine_hypothesis.add_edge("return to debate", "care_agent debater")
generate_and_refine_hypothesis.add_edge("return to debate", "justice_agent debater")
generate_and_refine_hypothesis.add_edge("return to debate", "utilitarian_agent debater")
generate_and_refine_hypothesis.add_edge("return to debate", "common_good_agent debater")

#compile
generate_and_refine_hypothesis = generate_and_refine_hypothesis.compile()

#display workflow
#display(Image(refine_hypothesis.get_graph().draw_mermaid_png()))

#invoke workflow
state = generate_and_refine_hypothesis.invoke(state)

print("Revised ToM hypothesis")
print("----------------------")
print(state['tom_hypothesis'])
print("----------------------")

save_state_to_file(state)