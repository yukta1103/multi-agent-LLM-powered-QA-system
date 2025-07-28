import json

class VerifierAgent:
    def __init__(self):
        self.prev_state = None

    def _parse_ui_tree(self, obs):
        ui_json = obs["pixels"]["ui_tree"]
        if isinstance(ui_json, bytes):
            ui_json = ui_json.decode("utf-8")
        try:
            return json.loads(ui_json)
        except Exception as e:
            print("UI parse error:", e)
            return {}

    def verify(self, subgoal, obs):
        current_tree = self._parse_ui_tree(obs)
        print(f"\n[Verifier] Verifying subgoal: '{subgoal}'")

        if "toggle" in subgoal.lower() or "switch" in subgoal.lower():

            for child in current_tree.get("children", []):
                if "switch" in child.get("text", "").lower():
                    state = child.get("state", "UNKNOWN").lower()
                    print(f"[Verifier] Toggle state: {state}")
                    if "on" in subgoal.lower() and state == "on":
                        print("[Verifier] Wi-Fi successfully turned ON.")
                        return "pass"
                    elif "off" in subgoal.lower() and state == "off":
                        print("[Verifier] Wi-Fi successfully turned OFF.")
                        return "pass"
                    else:
                        print("[Verifier] Toggle state did not match subgoal.")
                        return "fail"

        if subgoal.lower() in json.dumps(current_tree).lower():
            print("[Verifier] Subgoal text found in UI.")
            return "pass"
        else:
            print("[Verifier] Subgoal not reflected in UI.")
            return "fail"
