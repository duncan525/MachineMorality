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

    ### USER PROMPT ###
    with open("user_info/user_prompt.txt", "r") as f:
        state_data['user_prompt'] = f.read()

    ### TOM HYPOTHESIS ###
    state_data['tom_hypothesis'] = ""

    ### S0 SCHEMA TOKEN COUNTS ###
    care_agent_schema_tokens = 0
    justice_agent_schema_tokens = 0
    utilitarian_agent_schema_tokens = 0
    common_good_agent_schema_tokens = 0

    ### S1 HYPOTHESIS COMPONENETS ###
    state_data['beliefs_hypothesis'] = ""
    state_data['emotions_hypothesis'] = ""
    state_data['motives_hypothesis'] = ""
    state_data['knowledge_hypothesis'] = ""

    ### S2 DEBATE ROUND LIMITS ###
    state_data['max_rounds'] = 3
    state_data['rounds_performed'] = 0

    ### S1 MOST DISAGREED COMPONENT ###
    state_data['worst_component'] = ""

    ### S1 DEBATE STEP ###
    state_data['debate_step'] = 1

    ### S1 HYPOTHESES ###
    state_data['care_beliefs'] = ""
    state_data['care_emotions'] = ""
    state_data['care_motives'] = ""
    state_data['care_knowledge'] = ""

    state_data['justice_beliefs'] = ""
    state_data['justice_emotions'] = ""
    state_data['justice_motives'] = ""
    state_data['justice_knowledge'] = ""

    state_data['utilitarian_beliefs'] = ""
    state_data['utilitarian_emotions'] = ""
    state_data['utilitarian_motives'] = ""
    state_data['utilitarian_knowledge'] = ""

    state_data['common_good_beliefs'] = ""
    state_data['common_good_emotions'] = ""
    state_data['common_good_motives'] = ""
    state_data['common_good_knowledge'] = ""

    ### S1 AGENT FEEDBACK ###
    state_data['care_agent_feedback'] = []
    state_data['justice_agent_feedback'] = []
    state_data['utilitarian_agent_feedback'] = []
    state_data['common_good_agent_feedback'] = []

    ### S1 AGENT SCORES/VOTES ###
    str_list = ["care_agent_beliefs_votes", "care_agent_emotions_votes",
                "care_agent_motives_votes", "care_agent_knowledge_votes",
                "justice_agent_beliefs_votes", "justice_agent_emotions_votes",
                "justice_agent_motives_votes", "justice_agent_knowledge_votes",
                "utilitarian_agent_beliefs_votes", "utilitarian_agent_emotions_votes",
                "utilitarian_agent_motives_votes", "utilitarian_agent_knowledge_votes",
                "common_good_agent_beliefs_votes", "common_good_agent_emotions_votes",
                "common_good_agent_motives_votes", "common_good_agent_knowledge_votes"]
    
    for string in str_list:
        state_data[string] = ""

    ### S1 COMPONENT SCORES ###
    state_data['beliefs_score'] = 0
    state_data['emotions_score'] = 0
    state_data['motives_score'] = 0
    state_data['knowledge_score'] = 0

    ### S1 DEBATE JUDGE DECISION ###
    state_data['judgment'] = ""

    ### S2 AGENT SCORES ###
    state_data['care_agent_score'] = 0
    state_data['justice_agent_score'] = 0
    state_data['utilitarian_agent_score'] = 0
    state_data['common_good_agent_score'] = 0

    ### S2 METACOGNITIVE LOOP LIMITS ###
    state_data['max_loops'] = 3
    state_data['loops_performed'] = 0

    ### S2 RESPONSE PROMPT ###
    #to allow benchmark testing without extraneous instructions
    with open("user_info/response_prompt.txt", "r") as f:
        response_prompt = f.read()

    if response_prompt == "":
        state_data['response_prompt'] = state_data['user_prompt']
    else:
        state_data['response_prompt'] = response_prompt

    ### S2 FINAL RESPONSE ###
    state_data['final_response'] = ""
    
    return state_data
    
state: State = initialize_state()

#save info
def save_state_to_file(state: State):
    updated_state = json.dumps(state, indent=4)

    with open("states/stage0_state.txt", "w", encoding="utf-8") as f:
        f.write(updated_state) 

## AGENT

#output schemas
class CareSchema(BaseModel): ## ethics schema for care agent
    relationships: str          #what relationships exist between the
                                    #parties? in what ways are they
                                    #morally significant?
    needs: str                  #what does each party need?
    attentiveness: str          #how might each party be aware of the others'
                                    #needs?
    responsibility: str         #is each party willing to respond to & take
                                    #care of the others' needs? if not, why?
                                    #how might they become willing under
                                    #changing circumstances?
    competence: str             #is each party capable of providing good &
                                    #successful care to the others' needs?
    responsiveness: str         #how might each party consider others'
                                    #positions from the other's PoV? how
                                    #might they recognize potential for
                                    #abuse in care?

class JusticeSchema(BaseModel): ## ethics schema for justice agent
    conflicts_of_interest: str  #what conflicts of interest might exist or
                                    #arise due to the relative scarcity of
                                    #goods/services? how might each party
                                    #disagree over who gets what?
    dignity: str                #how can each party recognize & maintain
                                    #each other's dignity?
    dependencies: str           #how do the parties depend on one another?
    instability: str            #how might the stability of the
                                    #dependencies be threatened by an
                                    #unjust solution?
    distribution: str           #how should the relatively scarce goods/
                                    #services be distributed according to
                                    #principles of fairness & equity?

class UtilitarianSchema(BaseModel): ## ethics schema for utilitarian agent
    potential_actions: str      #what plausible courses of action are
                                    #available in this situation?
    potential_benefits: str     #how could any of my potential actions
                                    #benefit the parties (& lead to just
                                    #outcomes?) and anyone else who could
                                    #be affected?
    potential_harms: str        #how could any of my potential actions
                                    #harm the parties and anyone else who
                                    #could be affected?
    values: str                 #how are the benefits & harms for each
                                    #potential action valued against those
                                    #of other potential actions?
    balancing_consequences: str #how can we balance the potential benefits
                                    #and harms of taking each potential
                                    #action?

class CommonGoodSchema(BaseModel): ## ethics schema for common good agent
    general_conditions: str     #what general conditions would be to all
                                    #parties' advantage?
    particular_conditions: str  #what particular conditions would be to
                                    #the advantage of each party?
    ubiquitous_goods: str       #what goods can all parties access?
    benefits: str               #how might each party benefit by
                                    #maintaining those goods/conditions
                                    #all parties can access?
    costs: str                  #what costs might each party bear to
                                    #maintain those goods/conditions that
                                    #all parties can access & benefit from?

class FrameworkSchema(BaseModel): ## ethics schema for s1 debate orchestrator & s2 response agent
    rights: str                 #what rights do people in general possess?
                                    #how do those rights apply to each
                                    #party? how can those rights best be
                                    #respected?
    justice: str                #what does each party deserve according to
                                    #principles of justice & fairness? what
                                    #dependencies exist between them? how
                                    #might those dependencies be threatened
                                    #by an unfair/inequitable solution?
    utilitarianism: str         #what courses of action are available? how
                                    #could they benefit and/or harm the
                                    #affected parties? what action
                                    #produces the best net consequences
                                    #based on the benefits/harms of each
                                    #action?
    common_good: str            #what general conditions would be to all
                                    #parties' advantage? what goods can
                                    #all parties access? what sorts of
                                    #costs and benefits would each party
                                    #incur by maintaining those goods/
                                    #conditions?
    virtues: str                #what virtues should each party exhibit?
                                    #how can each party act on those
                                    #virtues? in what ways does each party
                                    #fail to exemplify virtuous behavior,
                                    #and how could they improve?
    care: str                   #what relationships exist between the
                                    #parties? how are they morally
                                    #significant? does each party have the
                                    #awareness, willingness, and ability
                                    #necessary to care for one another?
    

#parsers
care_parser = PydanticOutputParser(
    pydantic_object = CareSchema
)

justice_parser = PydanticOutputParser(
    pydantic_object = JusticeSchema
)

utilitarian_parser = PydanticOutputParser(
    pydantic_object = UtilitarianSchema
)

common_good_parser = PydanticOutputParser(
    pydantic_object = CommonGoodSchema
)

framework_parser = PydanticOutputParser(
    pydantic_object = FrameworkSchema
)

#tools
care_tools = [sep_search, iep_search, britannica_search, ask_philosophers_search, 
              philosophers_magazine_search, rep_search, care_ethics_lens]
justice_tools = [sep_search, iep_search, britannica_search, ask_philosophers_search, 
              philosophers_magazine_search, rep_search, justice_lens]
utilitarian_tools = [sep_search, iep_search, britannica_search, ask_philosophers_search, 
              philosophers_magazine_search, rep_search, utilitarian_lens]
common_good_tools = [sep_search, iep_search, britannica_search, ask_philosophers_search, 
              philosophers_magazine_search, rep_search, common_good_lens]
framework_tools = [sep_search, iep_search, britannica_search, ask_philosophers_search, 
              philosophers_magazine_search, rep_search, ethical_decision_framework,
              rights_lens, justice_lens, utilitarian_lens, common_good_lens, virtues_lens,
              care_ethics_lens]

#system prompts
care_system_prompt = f"""
                   You are an AI reasoning agent that performs
                   ethical analysis. You have an
                   ethical perspective summarized in an "ethics
                   schema". You will respond to queries related
                   to the user prompt. You *must* use the provided
                   tools ({care_tools}) as you respond to the
                   query. Think step-by-step before
                   answering. Return ONLY valid JSON in the
                   format specified by: {care_parser.get_format_instructions()}
                """

justice_system_prompt = f"""
                   You are an AI reasoning agent that performs
                   ethical analysis. You have an
                   ethical perspective summarized in an "ethics
                   schema". You will respond to queries related
                   to the user prompt. You *must* use the provided
                   tools ({justice_tools}) as you respond to the
                   query. Think step-by-step before
                   answering. Return ONLY valid JSON in the
                   format specified by: {justice_parser.get_format_instructions()}
                """

utilitarian_system_prompt = f"""
                   You are an AI reasoning agent that performs
                   ethical analysis. You have an
                   ethical perspective summarized in an "ethics
                   schema". You will respond to queries related
                   to the user prompt. You *must* use the provided
                   tools ({utilitarian_tools}) as you respond to the
                   query. Think step-by-step before
                   answering. Return ONLY valid JSON in the
                   format specified by: {utilitarian_parser.get_format_instructions()}
                """

common_good_system_prompt = f"""
                   You are an AI reasoning agent that performs
                   ethical analysis. You have an
                   ethical perspective summarized in an "ethics
                   schema". You will respond to queries related
                   to the user prompt. You *must* use the provided
                   tools ({common_good_tools}) as you respond to the
                   query. Think step-by-step before
                   answering. Return ONLY valid JSON in the
                   format specified by: {common_good_parser.get_format_instructions()}
                """

framework_system_prompt = f"""
                        You are an AI reasoning agent that performs
                        ethical analysis. You have an
                        ethical perspective summarized in an "ethics
                        schema". You will respond to queries related
                        to the user prompt. You *must* use the provided
                        tools ({framework_tools}) as you respond to the
                        query. Think step-by-step before
                        answering. Return ONLY valid JSON in the
                        format specified by: {framework_parser.get_format_instructions()}
                    """

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

#limit excessive tool calls (qwen3.5:4b, for example, loves querying tools 20+ times)
tool_call_limiter = ToolCallLimitMiddleware(
    run_limit = 5 #limits tool calls to 5 per prompt
    #exit behavior is "continue" by default
)
 

#allow retries
care_retry_parser = OutputFixingParser.from_llm(llm = llm, parser = care_parser, max_retries = 2)
justice_retry_parser = OutputFixingParser.from_llm(llm = llm, parser = justice_parser, max_retries = 2)
utilitarian_retry_parser = OutputFixingParser.from_llm(llm = llm, parser = utilitarian_parser, max_retries = 2)
common_good_retry_parser = OutputFixingParser.from_llm(llm = llm, parser = common_good_parser, max_retries = 2)
framework_retry_parser = OutputFixingParser.from_llm(llm = llm, parser = framework_parser, max_retries = 2)

#agents
care_agent = create_agent(model = llm,
                     tools = care_tools,
                     system_prompt = care_system_prompt,
                     middleware = [tool_call_limiter]
                    )

justice_agent = create_agent(model = llm,
                     tools = justice_tools,
                     system_prompt = justice_system_prompt,
                     middleware = [tool_call_limiter]
                    )

utilitarian_agent = create_agent(model = llm,
                     tools = utilitarian_tools,
                     system_prompt = utilitarian_system_prompt,
                     middleware = [tool_call_limiter]
                    )

common_good_agent = create_agent(model = llm,
                     tools = common_good_tools,
                     system_prompt = common_good_system_prompt,
                     middleware = [tool_call_limiter]
                    )

framework_agent = create_agent(model = llm,
                     tools = framework_tools,
                     system_prompt = framework_system_prompt,
                     middleware = [tool_call_limiter]
                    )

#parse_output function
def parse_output(state: State, parser, response) -> str:
    #parse structured output
    try:
        structured_response = parser.parse(
            response
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

#create 5 separate & unique belief schema
agents = [care_agent, justice_agent, utilitarian_agent, common_good_agent, framework_agent]

for i in range(len(agents)):
    agent = agents[i]
    schema = ""

    match i:
        case 0:
            retry_parser = care_retry_parser
            schema = "care_schema"
            prompt = f"""
                You approach ethics using the "care ethics" lens,
                but you are happy to listen to other perspectives.
                The user prompt is {state['user_prompt']}.

                As an ethical reasoning agent, you must determine how to
                apply your ethical perspective to the situation
                described in the user prompt. Your ethics schema
                is incomplete right now, so you must fill it in
                to determine how to apply it to this situation.
                Determine, as a care ethics reasoning agent, how your
                knowledge of the care ethics principles of
                relationships, needs, attentiveness,
                responsibility, competence, and responsiveness
                applies to the situation in the user prompt.
                The following document describes
                those components of care ethics in detail: 
                [{care_document}]
                
                Base your answers on the search results from the search
                tool(s) you used. Return ONLY valid JSON in the format specified by:
                {care_parser.get_format_instructions()}
                """
        case  1:
            retry_parser = justice_retry_parser
            schema = "justice_schema"
            prompt = f"""
                You approach ethics using the "justice" lens,
                but you are happy to listen to other perspectives.
                The user prompt is {state['user_prompt']}.=

                As an ethical reasoning agent, you must determine how to
                apply your ethical perspective to the situation
                described in the user prompt. Your ethics schema
                is incomplete right now, so you must fill it in
                to determine how to apply it to this situation.
                Determine, as a justice ethics reasoning agent, how your
                knowledge of the justice ethics principles
                of conflicts_of_interest, dignity, dependencies,
                instability, and distribution applies to the situation
                in the user prompt. The following document describes
                those components of justice ethics in detail:
                [{justice_document}]
                
                Base your answers on the search results from the search
                tool(s) you used. Return ONLY valid JSON in the format specified by:
                {justice_parser.get_format_instructions()}  
                """
        case 2:
            retry_parser = utilitarian_retry_parser
            schema = "utilitarian_schema"
            prompt = f"""
                You approach ethics using the "utilitarian" lens,
                but you are happy to listen to other perspectives.
                The user prompt is {state['user_prompt']}.

                As an ethical reasoning agent, you must determine how to
                apply your ethical perspective to the situation
                described in the user prompt. Your ethics schema
                is incomplete right now, so you must fill it in
                to determine how to apply it to this situation.
                Determine, as a utilitarian ethics reasoning agent, how your
                knowledge of the utilitarian ethics principles
                of potential_actions,
                potential_benefits, potential_harms, values, and
                balancing_consequences applies to the situation in
                the user prompt. The following document describes
                those components of utilitarian ethics in detail:
                [{utilitarian_document}]
                
                Base your answers on the search results from the search
                tool(s) you used. Return ONLY valid JSON in the format specified by:
                {utilitarian_parser.get_format_instructions()}
                """
        case 3:
            retry_parser = common_good_retry_parser
            schema = "common_good_schema"
            prompt = f"""
                You approach ethics using the "common_good" lens,
                but you are happy to listen to other perspectives.
                The user prompt is {state['user_prompt']}.

                As an ethical reasoning agent, you must determine how to
                apply your ethical perspective to the situation
                described in the user prompt. Your ethics schema
                is incomplete right now, so you must fill it in
                to determine how to apply it to this situation.
                Determine, as a common good ethics reasoning agent, how
                your knowledge of the common good ethics
                principles of general_conditions, particular_conditions,
                ubiquitous_goods, benefits, and costs applies to the
                situation in the user prompt. The following document
                describes those components of common good ethics in
                detail: [{common_good_document}]
                
                Base your answers on the search results from the search
                tool(s) you used. Return ONLY valid JSON in the format specified by:
                {common_good_parser.get_format_instructions()}
            """
        case 4:
            retry_parser = framework_retry_parser
            schema = "framework_schema"
            prompt = f"""
                As an ethical reasoning agent, you approach ethics using
                a rigorous framework for ethical decision-making.
                Your role is to consider several different and
                potentially conflicting ethical perspectives/lenses and
                resolve them into a single, coherent decision that pulls from each
                lens as it is relevant.
                
                The user prompt is {state['user_prompt']}.
                Your ethics schema is incomplete right now, so you
                must fill it in to determine how to apply it to this
                situation. Determine, as an ethical reasoning agent, how
                your knowledge of the principles of the framework
                for ethical decision making (rights, justice,
                utilitarianism, common_good, virtues, and care)
                applies to the situation in the user prompt.
                The following document describes those components of 
                the framework in detail: [{framework_document}]
                
                Base your answers on the search results from the search
                tool(s) you used. Return ONLY valid JSON in the format specified by:
                {framework_parser.get_format_instructions()}
            """

    match i:
        case 0:
            print("care_agent responding to prompt...")
        case 1:
            print("justice_agent responding to prompt...")
        case 2:
            print("utilitarian_agent responding to prompt...")
        case 3:
            print("common_good_agent responding to prompt...")
        case 4:
            print("framework_agent responding to prompt...")
    
    raw_response = agent.invoke(
            { "messages": [{"role": "system", "content": prompt}] }
        )

    messages = raw_response["messages"]

    #log tool calls if they happened
    log.tool_calls(messages)
    
    response = messages[-1].content
    structured_response = parse_output(state, retry_parser, response)

    match i:
        case 0:
            print(f"care_agent responded to prompt with structured response:")
        case 1:
            print(f"justice_agent responded to prompt with structured response:")
        case 2:
            print(f"utilitarian_agent responded to prompt with structured response:")
        case 3:
            print(f"common_good_agent responded to prompt with structured response:")
    print(structured_response)
    print()
            
    #write belief schema to file
    with open(f"states/schemas/{schema}.txt", "w", encoding="utf-8") as f:
        f.write(str(structured_response))

save_state_to_file(state)
