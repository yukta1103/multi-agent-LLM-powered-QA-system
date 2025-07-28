import os
from openai import OpenAI

client = OpenAI(api_key="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") 

class PlannerAgent:
    def plan(self, goal, state):
        prompt = f"Given the current environment state:\n{state}\nPlan steps to achieve goal:\n{goal}\nReturn step-by-step actions."
        plan_steps = self.query_llm_for_plan(prompt)
        return plan_steps

    def query_llm_for_plan(self, prompt):
        response = llm_api_call(prompt)
        steps = parse_steps_from_response(response)
        return steps

if __name__ == "__main__":
    agent = PlannerAgent()
    subgoals = agent.plan("Test turning Wi-Fi on and off")
    for step in subgoals:
        print("•", step)
