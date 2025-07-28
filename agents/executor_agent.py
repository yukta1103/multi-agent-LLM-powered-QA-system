import json

class MockAndroidEnv:
    def __init__(self):
        self.state = "wifi_closed"
        self.obs = self.reset()  

    def reset(self):
        print("[MOCK ENV] Resetting environment.")
        self.state = "wifi_closed"
        self.obs = self._get_obs()
        return self.obs

    def step(self, action):
        print(f"[MOCK ENV] Executing action: {action}")
        if action["element_id"] == "wifi_text":
            self.state = "wifi_screen_open"
        elif action["element_id"] == "wifi_toggle":
            if self.state == "wifi_screen_open":
                self.state = "wifi_on" if self.state != "wifi_on" else "wifi_off"
        self.obs = self._get_obs() 
        return self.obs, 1, False, {}

    def perform_action(self, step):
        step_lower = step.lower()
        if "tap on wi-fi" in step_lower:
            action = {"action_type": "touch", "element_id": "wifi_text"}
        elif "toggle wi-fi switch on" in step_lower:
            action = {"action_type": "touch", "element_id": "wifi_toggle"}
        else:
            print(f"[MOCK ENV] Unknown step: {step}")
            return None
        return self.step(action)

    def get_state(self):
        return self.state

    def _get_obs(self):
        toggle_state = "ON" if self.state == "wifi_on" else "OFF"
        root_text = "Wi-Fi Settings" if self.state == "wifi_screen_open" else "Wi-Fi"

        ui_tree = {
            "text": root_text,
            "resource_id": "wifi_text",
            "children": [
                {
                    "text": "Wi-Fi switch",
                    "resource_id": "wifi_toggle",
                    "state": toggle_state
                }
            ]
        }
        return {"pixels": {"ui_tree": json.dumps(ui_tree)}}

    def check_expected_state_after_action(self, step):
        step_lower = step.lower()
        if "tap on wi-fi" in step_lower:
            return self.state == "wifi_screen_open"
        elif "toggle wi-fi switch on" in step_lower:
            return self.state == "wifi_on"
        return False

class PlannerAgent:
    def plan(self, goal, state):
        print(f"[Planner] Planning for goal: '{goal}' with current state: {state}")
        if goal.lower() == "tap on wi-fi":
            return ["Tap on Wi-Fi"]
        elif goal.lower() == "toggle wi-fi switch on":
            return ["Tap on Wi-Fi", "Toggle Wi-Fi switch ON"]
        else:
            print("[Planner] No plan found for goal.")
            return []

class ExecutorAgent:
    def __init__(self, planner_agent, environment):
        self.planner_agent = planner_agent
        self.env = environment
        self.current_plan = []
        self.current_goal = None

    def execute_plan(self, plan, goal):
        self.current_plan = plan
        self.current_goal = goal

        max_replans = 5
        replan_count = 0
        step_index = 0

        while step_index < len(self.current_plan):
            step = self.current_plan[step_index]
            print(f"Executing step {step_index + 1}/{len(self.current_plan)}: {step}")

            success = self.execute_step(step)
            if not success:
                print(f"Step '{step}' failed. Triggering replanning...")
                replan_count += 1
                if replan_count > max_replans:
                    print("Exceeded max replanning attempts. Aborting execution.")
                    return False

                current_state = self.get_environment_state()
                new_plan = self.planner_agent.plan(goal=self.current_goal, state=current_state)
                print(f"[Executor] New plan after replanning: {new_plan}")
                
                if not new_plan or new_plan == self.current_plan:
                    print("Replanning did not produce a new plan or same as current. Aborting to avoid infinite loop.")
                    return False

                self.current_plan = new_plan
                step_index = 0
                continue

            step_index += 1

        print("Plan executed successfully.")
        return True


    def execute_step(self, step):
        try:
            action_result = self.env.perform_action(step)
            if not self.verify_effect(step):
                print(f"Effect verification failed for step: {step}")
                return False
            return True
        except Exception as e:
            print(f"Exception during step execution: {e}")
            return False

    def verify_effect(self, step):
        return self.env.check_expected_state_after_action(step)

    def get_environment_state(self):
        return self.env.get_state()

    def execute_subgoal(self, subgoal):
        ui_tree = self._get_ui_tree()
        print("\n[Executor] Subgoal:", subgoal)
        element = self._find_element_by_text(ui_tree, subgoal)
        if element and "resource_id" in element:
            action = {
                "action_type": "touch",
                "element_id": element["resource_id"]
            }
            self.obs, reward, done, info = self.env.step(action)
        else:
            print("Element not found for:", subgoal)

class VerifierAgent:
    def __init__(self):
        pass

    def _parse_ui_tree(self, obs):
        ui_json = obs["pixels"]["ui_tree"]
        if isinstance(ui_json, bytes):
            ui_json = ui_json.decode("utf-8")
        return json.loads(ui_json)

    def verify(self, subgoal, obs):
        current_tree = self._parse_ui_tree(obs)
        print(f"\n[Verifier] Verifying subgoal: '{subgoal}'")

        if "toggle" in subgoal.lower():
            for child in current_tree.get("children", []):
                if "switch" in child.get("text", "").lower():
                    state = child.get("state", "").lower()
                    if "on" in subgoal.lower() and state == "on":
                        print("[Verifier] Wi-Fi successfully turned ON.")
                        return "pass"
                    elif "off" in subgoal.lower() and state == "off":
                        print("[Verifier] Wi-Fi successfully turned OFF.")
                        return "pass"
                    else:
                        print("[Verifier] Toggle state did not match subgoal.")
                        return "fail"

        if "tap on wi-fi" in subgoal.lower():
            if "Wi-Fi Settings" in json.dumps(current_tree):
                print("[Verifier] Wi-Fi screen successfully opened.")
                return "pass"
            else:
                print("[Verifier] Wi-Fi screen not opened.")
                return "fail"

        print("[Verifier] Subgoal not verified.")
        return "fail"

if __name__ == "__main__":
    env = MockAndroidEnv()
    planner = PlannerAgent()
    executor = ExecutorAgent(planner_agent=planner, environment=env)
    verifier = VerifierAgent()

    test_goals = [
        "Tap on Wi-Fi",
        "Toggle Wi-Fi switch ON"
    ]

    for goal in test_goals:
        print(f"\n[Main] Starting execution for goal: '{goal}'")
        plan = planner.plan(goal, env.get_state())
        executor.execute_plan(plan, goal)
        result = verifier.verify(goal, executor.env.obs)
        print("[Result]:", result)
