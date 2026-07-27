from dotenv import load_dotenv
from pydantic import BaseModel
from datetime import datetime

### STATE ###
from typing_extensions import TypedDict #allow for typed state
from typing import List, Annotated #important for state & state updates
import operator #important for state updates

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

    with open("states/stage1_state.txt", "r") as f:
        state_data = json.load(f)

    return state_data
    
state: State = initialize_state()

#save info
def save_state_to_file(state: State):
    updated_state = json.dumps(state, indent=4)

    with open("states/stage2_state.txt", "w", encoding="utf-8") as f:
        f.write(updated_state)

## AGENT

#output schema
class ResponseSchema(BaseModel):
    response: str

class ScoreSchema(BaseModel):
    score: int
    feedback: str

#parser
response_parser = PydanticOutputParser(
    pydantic_object = ResponseSchema
)

score_parser = PydanticOutputParser(
    pydantic_object = ScoreSchema
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
    temperature = temperature,
    max_completion_tokens = 300,
    timeout = 30,
    max_retries = 3
)

#allow retries
response_retry_parser = OutputFixingParser.from_llm(llm = llm, parser = response_parser, max_retries = 2)
score_retry_parser = OutputFixingParser.from_llm(llm = llm, parser = score_parser, max_retries = 2)

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

# Nodes

def respond(state: State):
    print("generating final response...")

    prompt = ""
    if state['final_response'] == "":
        prompt = f"""
            User prompt: [{state['response_prompt']}]. Hypothesis about the state of
            mind of the subject of the user prompt: [{state['tom_hypothesis']}].
            
            Based on the user prompt and the hypothesis about its subject's state of mind,
            generate a response to the user prompt that adheres to your own ethical
            principles and that incorporates those expressed in the hypothesis. Make sure
            your response follows any formatting instructions in the user prompt.
            Use the hypothesis' understanding of the subject's state of mind to guide
            your response. It is very important that your response attends to the
            subject's mental state in the way the hypothesis describes it.
            
            Pay special attention to any formatting instructions in the user prompt -- you
            *must* follow them, if there are any. Return ONLY valid JSON in the format
            specified by: {response_parser.get_format_instructions()} """
    else:
        feedback = state['care_agent_feedback'][-1] + " " + state['justice_agent_feedback'][-1] + " " +\
            state['utilitarian_agent_feedback'][-1] + " " + state['common_good_agent_feedback'][-1]
        
        prompt = f"""
            User prompt: [{state['response_prompt']}]. Hypothesis about the user's
            state of mind: [{state['tom_hypothesis']}]. Your previous response to
            the user prompt: [{state['final_response']}]. Feedback from other reasoning
            agents specializing in various types of ethics to your previous response: [{feedback}].

            Based on the user prompt, the hypothesis about its subject's state of mind,
            your previous response to the user prompt, and the feedback from other
            reasoning agents, generate a new and improved response to the user prompt.
            Your response must not only adhere to your own ethical principles and those
            expressed in the hypothesis, but it must also address the concerns about
            your previous response that were raised in the other ethicists' feedback.
            Use the hypothesis' understanding of the subject's state of mind to guide
            your response. It is very important that your response attends to the
            subject's mental state in the way the hypothesis describes it.
                         
            Pay special attention to any formatting instructions in the user prompt -- you
            *must* follow them, if there are any, regardless of the feedback. Return ONLY
            valid JSON in the format specified by: {response_parser.get_format_instructions()}"""

    raw_response = framework_agent.invoke(
            {"messages": [{"role": "system", "content": prompt}]}
        )

    messages = raw_response["messages"]
    response = messages[-1].content

    #parse structured output
    structured_response = parse_output(state, response_retry_parser, response)

    print("final response")
    print("--------------")
    print(structured_response)
    print("--------------")
    print()

    return {"final_response": structured_response[10:-1]} #cut out response=''

def care_agent_score(state: State):
    """ quantify ethical soundness of response """

    print("generating care_agent score...")
    
    prompt = f"""User prompt: [{state['user_prompt']}]. Response to the user prompt
                written by another reasoning agent: [{state['final_response']}]. 
                          
                Score the response to the user prompt out of 10 based on how well it
                generally considers and incorporates the ethical principles of your ethical lens. The response does not have to focus on your perspective exclusively,
                and it does not need to incorporate every principle to the letter -- it just
                needs to get the general idea and utilize care ethics effectively.
                Additionally, give feedback on how the response could improve.
                Be critical; regardless of the score you give, your feedback should focus
                on how the other reasoning agent could improve when they rewrite their
                response to the user prompt. Keep your feedback brief (under 50 words)
                and to-the-point; focus on the most important points of improvement.

                For the score component, return a single number in [1, 10]. For the
                feedback component, generate specific, actionable feedback on how the
                response can remedy its failures to properly consider the principles
                of care ethics.
                          
                Return ONLY valid JSON in the format specified by:
                {score_parser.get_format_instructions()}"""

    #invoke agent
    raw_response = care_agent.invoke(
            {"messages": [{"role": "system", "content": prompt}]}
            )

    messages = raw_response["messages"]
    response = messages[-1].content

    #parse structured output
    structured_response = parse_output(state, score_retry_parser, response)

    #store response as dictionary
    hypothesis = {}
    feedback_index = structured_response.find("feedback=")

    #cut out stuff like "feedback=" and extraneous apostrophes
    hypothesis['score'] = structured_response[6:(feedback_index-1)]
    hypothesis['feedback'] = structured_response[(feedback_index+10):-1]

    print("care_agent score + feedback")
    print("---------------------------")
    print(structured_response)
    print("---------------------------")
    print()

    try:
        score = float(hypothesis['score'])
    except:
        score = 0

    return {"care_agent_score": score,
            "care_agent_feedback": [hypothesis['feedback']]}

def justice_agent_score(state: State):
    """ quantify ethical soundness of response """

    print("generating justice_agent score...")

    prompt = f"""User prompt: [{state['user_prompt']}]. Response to the user prompt
                written by another reasoning agent: [{state['final_response']}]. 
                          
                Score the response to the user prompt out of 10 based on how well it
                generally considers and incorporates the ethical principles of your ethical lens. The response does not have to focus on your perspective exclusively,
                and it does not need to incorporate every principle to the letter -- it just
                needs to get the general idea and utilize care ethics effectively.
                Additionally, give feedback on how the response could improve.
                Be critical; regardless of the score you give, your feedback should focus
                on how the other reasoning agent could improve when they rewrite their
                response to the user prompt. Keep your feedback brief (under 50 words)
                and to-the-point; focus on the most important points of improvement.

                For the score component, return a single number in [1, 10]. For the
                feedback component, generate specific, actionable feedback on how the
                response can remedy its failures to properly consider the principles
                of justice ethics.
                          
                Return ONLY valid JSON in the format specified by:
                {score_parser.get_format_instructions()}"""

    #invoke agent
    raw_response = justice_agent.invoke(
            {"messages": [{"role": "system", "content": prompt}]}
            )

    messages = raw_response["messages"]
    response = messages[-1].content

    #parse structured output
    structured_response = parse_output(state, score_retry_parser, response)

    #store response as dictionary
    hypothesis = {}
    feedback_index = structured_response.find("feedback=")

    #cut out stuff like "feedback=" and extraneous apostrophes
    hypothesis['score'] = structured_response[6:(feedback_index-1)]
    hypothesis['feedback'] = structured_response[(feedback_index+10):-1]

    print("justice_agent score + feedback")
    print("---------------------------")
    print(structured_response)
    print("---------------------------")
    print()

    try:
        score = float(hypothesis['score'])
    except:
        score = 0

    return {"justice_agent_score": score,
            "justice_agent_feedback": [hypothesis['feedback']]}

def utilitarian_agent_score(state: State):
    """ quantify ethical soundness of response """

    print("generating utilitarian_agent score...")

    prompt = f"""User prompt: [{state['user_prompt']}]. Response to the user prompt
                written by another reasoning agent: [{state['final_response']}]. 
                          
                Score the response to the user prompt out of 10 based on how well it
                generally considers and incorporates the ethical principles of your ethical lens. The response does not have to focus on your perspective exclusively,
                and it does not need to incorporate every principle to the letter -- it just
                needs to get the general idea and utilize care ethics effectively.
                Additionally, give feedback on how the response could improve.
                Be critical; regardless of the score you give, your feedback should focus
                on how the other reasoning agent could improve when they rewrite their
                response to the user prompt. Keep your feedback brief (under 50 words)
                and to-the-point; focus on the most important points of improvement.

                For the score component, return a single number in [1, 10]. For the
                feedback component, generate specific, actionable feedback on how the
                response can remedy its failures to properly consider the principles
                of utilitarian ethics.
                          
                Return ONLY valid JSON in the format specified by:
                {score_parser.get_format_instructions()}"""

    #invoke agent
    raw_response = utilitarian_agent.invoke(
            {"messages": [{"role": "system", "content": prompt}]}
            )

    messages = raw_response["messages"]
    response = messages[-1].content

    #parse structured output
    structured_response = parse_output(state, score_retry_parser, response)

    #store response as dictionary
    hypothesis = {}
    feedback_index = structured_response.find("feedback=")

    #cut out stuff like "feedback=" and extraneous apostrophes
    hypothesis['score'] = structured_response[6:(feedback_index-1)]
    hypothesis['feedback'] = structured_response[(feedback_index+10):-1]

    print("utilitarian_agent score + feedback")
    print("---------------------------")
    print(structured_response)
    print("---------------------------")
    print()

    try:
        score = float(hypothesis['score'])
    except:
        score = 0

    return {"utilitarian_agent_score": score,
            "utilitarian_agent_feedback": [hypothesis['feedback']]}

def common_good_agent_score(state: State):
    """ quantify ethical soundness of response """

    print("generating common_good_agent score...")

    prompt = f"""User prompt: [{state['user_prompt']}]. Response to the user prompt
                written by another reasoning agent: [{state['final_response']}]. 
                          
                Score the response to the user prompt out of 10 based on how well it
                generally considers and incorporates the ethical principles of your ethical lens. The response does not have to focus on your perspective exclusively,
                and it does not need to incorporate every principle to the letter -- it just
                needs to get the general idea and utilize care ethics effectively.
                Additionally, give feedback on how the response could improve.
                Be critical; regardless of the score you give, your feedback should focus
                on how the other reasoning agent could improve when they rewrite their
                response to the user prompt. Keep your feedback brief (under 50 words)
                and to-the-point; focus on the most important points of improvement.

                For the score component, return a single number in [1, 10]. For the
                feedback component, generate specific, actionable feedback on how the
                response can remedy its failures to properly consider the principles
                of common good ethics.
                          
                Return ONLY valid JSON in the format specified by:
                {score_parser.get_format_instructions()}"""

    #invoke agent
    raw_response = common_good_agent.invoke(
            {"messages": [{"role": "system", "content": prompt}]}
            )

    messages = raw_response["messages"]
    response = messages[-1].content

    #parse structured output
    structured_response = parse_output(state, score_retry_parser, response)

    #store response as dictionary
    hypothesis = {}
    feedback_index = structured_response.find("feedback=")

    #cut out stuff like "feedback=" and extraneous apostrophes
    hypothesis['score'] = structured_response[6:(feedback_index-1)]
    hypothesis['feedback'] = structured_response[(feedback_index+10):-1]

    print("common_good_agent score + feedback")
    print("---------------------------")
    print(structured_response)
    print("---------------------------")
    print()

    try:
        score = float(hypothesis['score'])
    except:
        score = 0

    return {"common_good_agent_score": score,
            "common_good_agent_feedback": [hypothesis['feedback']]}

def advance_loops_performed(state: State):
    print("loops_performed is now", state['loops_performed'] + 1)
    print()

    return {"loops_performed": state['loops_performed'] + 1}

def return_to_scores(state: State):
    """ control flow node for conditionally moving
        from response agent to scoring agents """
    return

def check_loop_limit(state: State):
    """ determine if loop limit reached after response """

    if state['max_loops'] <= state['loops_performed']:
        print("max loops reached!")
        return "PASS"
    
    return "FAIL"

def check_total_score(state: State):
    """ determine if total score is >= 36/40 (36/40 == 0.9) """

    score_cutoff = 36

    print("assessing total score")
    print("---------------------")

    scores = [ state['care_agent_score'],
               state['justice_agent_score'],
               state['utilitarian_agent_score'],
               state['common_good_agent_score'] ]

    total_score = sum(scores)

    if total_score >= score_cutoff:
        print("Passed due to score of", total_score, ">=", score_cutoff)
        print()
        return "PASS"

    print("Failed due to score of", total_score, "<", score_cutoff)
    print()
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

# Build workflow
generate_response = StateGraph(State)

# Add nodes
generate_response.add_node("respond", respond)

generate_response.add_node("care_agent score", care_agent_score)
generate_response.add_node("justice_agent score", justice_agent_score)
generate_response.add_node("utilitarian_agent score", utilitarian_agent_score)
generate_response.add_node("common_good_agent score", common_good_agent_score)

generate_response.add_node("advance loops performed", advance_loops_performed)

generate_response.add_node("return to scores", return_to_scores)

# Add edges to connect nodes
generate_response.add_edge(START, "respond")

generate_response.add_edge("return to scores", "care_agent score")
generate_response.add_edge("return to scores", "justice_agent score")
generate_response.add_edge("return to scores", "utilitarian_agent score")
generate_response.add_edge("return to scores", "common_good_agent score")

generate_response.add_edge("care_agent score", "advance loops performed")
generate_response.add_edge("justice_agent score", "advance loops performed")
generate_response.add_edge("utilitarian_agent score", "advance loops performed")
generate_response.add_edge("common_good_agent score", "advance loops performed")

generate_response.add_conditional_edges(
        "advance loops performed", check_total_score, {"PASS": END, "FAIL": "respond"}
    )

generate_response.add_conditional_edges(
        "respond", check_loop_limit, {"PASS": END, "FAIL": "return to scores"}
    )

# Compile
generate_response = generate_response.compile()

# Show workflow
#display(Image(generate_response.get_graph().draw_mermaid_png()))

#Invoke workflow
state = generate_response.invoke(state)

print("FINAL RESPONSE")
print("--------------")
print(state['final_response'])
print("--------------")

save_state_to_file(state)
