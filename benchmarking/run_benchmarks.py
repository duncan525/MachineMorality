import sys
import os
import re
import random
sys.path.append('/path/to/parent/directory/')

import subprocess

import asyncio

import json

from LLM_Ethics_Benchmark.morals.evaluation.mfq_evaluator import MFQEvaluator
from LLM_Ethics_Benchmark.morals.evaluation.dilemmas_evaluator import DilemmasEvaluator
from LLM_Ethics_Benchmark.morals.evaluation.wvs_evaluator import WVSEvaluator

from LLM_Ethics_Benchmark.morals.instruments.mfq import MoralFoundationsQuestionnaire
from LLM_Ethics_Benchmark.morals.instruments.dilemmas import MoralDilemmasInstrument
from LLM_Ethics_Benchmark.morals.instruments.wvs import WorldValuesSurveyInstrument

from SocialBench import dataset as ds
from langchain_ollama import ChatOllama
from benchmark_step import run_pipeline
from datetime import datetime

#retrieve temperature value from file
with open("states/temperature.txt", "r") as f:
    temperature = f.read()

def get_dilemmas_prompt_from_question(question_id, dilemma_context):
    dilemmas = MoralDilemmasInstrument(data_path="../../LLM_Ethics_Benchmark/data/instruments/dilemmas.json")
    dilemma_id, q_id = question_id.split("-", 1)
    
    # Get the specific question using dilemma_id and q_id
    questions = dilemmas.get_questions_by_dilemma(dilemma_id)
    question = next((q for q in questions if q.get("id") == q_id), None)
    question = dilemma_context + " " + question['text']

    return question

async def evaluate_ecm(model: str, use_pipeline: bool, i: int):
    wvs = WorldValuesSurveyInstrument(data_path="../../LLM_Ethics_Benchmark/data/instruments/wvs.json")
    evaluator = WVSEvaluator(wvs)

    llm = ChatOllama(
        model = model,    
        temperature = temperature,
        max_completion_tokens = 300,
        timeout = 30,
        max_retries = 3
    )

    evaluation_results = []

    start_time = datetime.now()

    with open("../../LLM_Ethics_Benchmark/data/instruments/wvs.json", "r") as f:
        data = json.load(f)

    for domain in data["domains"]:
        for q in data["domains"][domain]["questions"]:
            question_id = q['id']
            response_prompt = wvs.get_prompt_for_question(question_id)

            if use_pipeline:
                ## must separate formatting instructions from rest of prompt
                ## to ensure instruction following in pipeline
                with open("user_info/response_prompt.txt", "w", encoding="utf-8") as f:
                    f.write(response_prompt)

                prompt = response_prompt.split("Provide")[0]
                response = run_pipeline(prompt, model)
            else:
                response = llm.invoke(response_prompt).content

            result = evaluator.evaluate_response(question_id, response)
            evaluation_results.append(result)

            try:
                ## store the result in an intermediate file
                intermediate_results_string = f"""{question_id}
                                                    ------------
                                                    OVERALL: {result['overall_alignment']}
                                                    valid: {result['is_valid_response']}
                                                    in range: {result['in_acceptable_range']}
                                                    reasoning: {result['reasoning_quality']}\n\n"""
            except:
                intermediate_results_string = f"""{question_id} went wrong\n"""

            try:
                with open(f"benchmarking/ethics_results/{use_pipeline}/intermediate/intermediate_ecm_{model}_{use_pipeline}", "x") as f:
                    f.write(intermediate_results_string)
            except FileExistsError:
                with open(f"benchmarking/ethics_results/{use_pipeline}/intermediate/intermediate_ecm_{model}_{use_pipeline}", "a", encoding = "utf-8") as f:
                    f.write(intermediate_results_string)
    
    domain_results = evaluator.calculate_domain_metrics(evaluation_results)
    overall_results = evaluator.calculate_overall_metrics(evaluation_results)

    if overall_results['total_questions'] == 0:
        valid_percentage = "N/A (invalid results)"
    else:
        valid_percentage = overall_results['valid_responses'] / overall_results['total_questions']

    end_time = datetime.now()
    time_elapsed = end_time - start_time

    try:
        ## store the results in a results file
        results_string = f"""OVERALL
                            -------
                                %valid: {valid_percentage}
                                avg alignment: {overall_results['avg_overall_alignment']}
                                avg reasoning: {overall_results['avg_reasoning_quality']}
                                %in range: {overall_results['acceptable_range_ratio']}
                                time elapsed: {time_elapsed}
                                iteration: {i}
                            DOMAIN
                            -------
                                {domain_results}"""
    except:
        results_string = "something went wrong"
    
    try:
        with open(f"benchmarking/ethics_results/{use_pipeline}/final/final_ecm_{model}_{use_pipeline}", "x") as f:
            f.write(results_string)
    except FileExistsError:
        with open(f"benchmarking/ethics_results/{use_pipeline}/final/final_ecm_{model}_{use_pipeline}", "a", encoding = "utf-8") as f:
            f.write(results_string)

async def evaluate_rqi(model: str, use_pipeline: bool, i: int):
    dilemmas = MoralDilemmasInstrument(data_path="../../LLM_Ethics_Benchmark/data/instruments/dilemmas.json")
    evaluator = DilemmasEvaluator(dilemmas)

    llm = ChatOllama(
        model = model,    
        temperature = temperature,
        max_completion_tokens = 300,
        timeout = 30,
        max_retries = 3
    )

    evaluation_results = []

    start_time = datetime.now()

    with open("../../LLM_Ethics_Benchmark/data/instruments/dilemmas.json", "r") as f:
        data = json.load(f)

    for dilemma in data["dilemmas"]:
        print("assessing dilemma", dilemma['id'])
        for q in dilemma["questions"]:
            question_id = dilemma['id'] + "-" + q['id']
            prompt = get_dilemmas_prompt_from_question(question_id, dilemma['description'])

            if use_pipeline:
                response = run_pipeline(prompt, model)
            else:
                response = llm.invoke(prompt).content

            result = evaluator.evaluate_response(question_id, response)
            evaluation_results.append(result)

            ## store the result in an intermediate file
            intermediate_results_string = f"""{question_id}
                                                ------------
                                                OVERALL: {result['overall_score']}
                                                valid: {result['is_valid_response']}
                                                sem_sim: {result['semantic_similarity']}
                                                crit_eval: {result['criteria_evaluations']}
                                                crit_satis: {result['criteria_satisfaction']}
                                                reasoning: {result['reasoning_score']}\n\n"""

            try:
                with open(f"benchmarking/ethics_results/{use_pipeline}/intermediate/intermediate_rqi_{model}_{use_pipeline}", "x") as f:
                    f.write(intermediate_results_string)
            except FileExistsError:
                with open(f"benchmarking/ethics_results/{use_pipeline}/intermediate/intermediate_rqi_{model}_{use_pipeline}", "a", encoding = "utf-8") as f:
                    f.write(intermediate_results_string)
    
    all_results = evaluator.calculate_aggregate_scores(evaluation_results)

    end_time = datetime.now()
    time_elapsed = end_time - start_time

    # store the results in a results file
    results_string = f"""OVERALL: {all_results['avg_overall_score']}
                         sem_sim: {all_results['avg_semantic_similarity']}
                         crit_satis: {all_results['avg_criteria_satisfaction']}
                         reasoning: {all_results['avg_reasoning_score']}
                         time elapsed: {time_elapsed}
                         iteration: {i}"""
    
    try:
        with open(f"benchmarking/ethics_results/{use_pipeline}/final/final_rqi_{model}_{use_pipeline}", "x") as f:
            f.write(results_string)
    except FileExistsError:
        with open(f"benchmarking/ethics_results/{use_pipeline}/final/final_rqi_{model}_{use_pipeline}", "a", encoding = "utf-8") as f:
            f.write(results_string)

async def evaluate_mfa(model: str, use_pipeline: bool, i: int):
    mfq = MoralFoundationsQuestionnaire(data_path="../../LLM_Ethics_Benchmark/data/instruments/mfq.json")
    evaluator = MFQEvaluator(mfq)

    llm = ChatOllama(
        model = model,    
        temperature = temperature,
        max_completion_tokens = 300,
        timeout = 30,
        max_retries = 3
    )

    evaluation_results = []

    start_time = datetime.now()

    with open("../../LLM_Ethics_Benchmark/data/instruments/mfq.json", "r") as f:
        data = json.load(f)

    for foundation in data["foundations"]:
        for questions in data["foundations"][foundation]:
            if questions == "relevance_questions" or questions == "agreement_questions":
                for q in data["foundations"][foundation][questions]:
                    question_id = q['id']
                    response_prompt = mfq.get_prompt_for_question(question_id)

                    if use_pipeline:
                        ## must separate formatting instructions from rest of prompt
                        ## to ensure instruction following in pipeline
                        with open("user_info/response_prompt.txt", "w", encoding="utf-8") as f:
                            f.write(response_prompt)

                        prompt = response_prompt.split("\n\n")[0]
                        response = run_pipeline(prompt, model)
                    else:
                        response = llm.invoke(response_prompt).content

                    result = evaluator.evaluate_response(question_id, response)
                    evaluation_results.append(result)

                    ## store the result in an intermediate file
                    intermediate_results_string = f"""{question_id}
                                                      ------------
                                                      valid: {result['is_valid_response']}
                                                      score: {result['alignment_score']}\n\n"""

                    try:
                        with open(f"benchmarking/ethics_results/{use_pipeline}/intermediate/intermediate_mfa_{model}_{use_pipeline}", "x") as f:
                            f.write(intermediate_results_string)
                        print("written to new file!")
                    except FileExistsError:
                        print("file already exists! writing to another")
                        with open(f"benchmarking/ethics_results/{use_pipeline}/intermediate/intermediate_mfa_{model}_{use_pipeline}", "a", encoding = "utf-8") as f:
                            f.write(intermediate_results_string)
                            print("successfully written to other file!")
    
    all_results = evaluator.calculate_foundation_alignment(evaluation_results)

    try:
        avg_score = all_results['care'] + all_results['fairness'] + all_results['loyalty'] + \
                    all_results['authority'] + all_results['sanctity']
    except:
        avg_score = 0

    avg_score /= 5

    end_time = datetime.now()
    time_elapsed = end_time - start_time

    ## store the results in a results file
    results_string = f"""AVG: {avg_score}
                         care/harm: {all_results['care']}
                         fairness/cheating: {all_results['fairness']}
                         loyalty/betrayal: {all_results['loyalty']}
                         authority/subversion: {all_results['authority']}
                         sanctity/degradation: {all_results['sanctity']}
                         time elapsed: {time_elapsed}
                         iteration: {i}"""
    
    try:
        with open(f"benchmarking/ethics_results/{use_pipeline}/final/final_mfa_{model}_{use_pipeline}", "x") as f:
            f.write(results_string)
    except FileExistsError:
        with open(f"benchmarking/ethics_results/{use_pipeline}/final/final_mfa_{model}_{use_pipeline}", "a", encoding = "utf-8") as f:
            f.write(results_string)

def run_ethics_benchmark(model):
    for use_pipeline in [True]:
        for i in range(3): #adjust for more or fewer trials
            print(f"Evaluating ECM for model {model} and use_pipeline {use_pipeline} and i={i}...")
            asyncio.run(evaluate_ecm(model, use_pipeline, i))
    
            with open("user_info/response_prompt.txt", "w", encoding="utf-8") as f:
                f.write("")
    
            print(f"Evaluating MFA for model {model} and use_pipeline {use_pipeline} and i={i}...")
            asyncio.run(evaluate_mfa(model, use_pipeline, i))
    
            with open("user_info/response_prompt.txt", "w", encoding="utf-8") as f:
                f.write("")
    
            print(f"Evaluating RQI for model {model} and use_pipeline {use_pipeline} and i={i}...")
            asyncio.run(evaluate_rqi(model, use_pipeline, i))
    
            with open("user_info/response_prompt.txt", "w", encoding="utf-8") as f:
                f.write("")

models = [ "gemma4:e4b", "gemma4:12b", "gemma4:26b" ]

if __name__ == "__main__":
    for model in models:
        run_ethics_benchmark(model)
