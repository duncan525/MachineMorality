import subprocess
import json
import ollama
user_prompt = input("Enter a prompt: ")

model = ""
while True:
    model = input("Enter a model or type quit to quit: ")
    
    if model == "quit":
        break

    try:
        ollama.show(model)
        to_break = True
    except:
        print("Invalid Input. Try again.")
        to_break = False
        
    if to_break:
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
