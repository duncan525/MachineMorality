import subprocess
import json
user_prompt = input("Enter a prompt: ")

model = ""
while True:
    model = input("Enter a model or type quit to quit:
    
    if model == "quit":
        break
    elif not ollama.exists(model):
        print("Invalid input. Try again.")
    else:
        break

if model != "quit":
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

    print("FINAL RESPONSE")
    print("----------------------")
    print(final_response)
    print("----------------------")
