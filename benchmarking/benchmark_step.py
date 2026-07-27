import json
import subprocess

def run_pipeline(user_prompt: str, model: str) -> str:
    with open("states/model.txt", "w", encoding="utf-8") as f:
        f.write(model)

    with open("user_info/user_prompt.txt", "w", encoding="utf-8") as f:
        f.write(user_prompt)

    #clear agent trajectory from last session
    with open("states/agent_trajectory.txt", "w", encoding="utf-8") as f:
        f.write("")

    print("running 's0_develop_schema.py'")
    subprocess.run(["python", "s0_develop_schema.py"])

    print("running 's1_develop_hypothesis.py'") 
    subprocess.run(["python", "s1_develop_hypothesis.py"])

    print("running 's2_generate_response.py'")   
    subprocess.run(["python", "s2_generate_response.py"])

    with open("states/stage2_state.txt", "r") as f:
        final_response = json.load(f)["final_response"]

    return final_response
