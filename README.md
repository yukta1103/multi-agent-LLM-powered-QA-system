# multi-agent-LLM-powered-QA-system


# Dynamic Replanning Executor Agent

## Overview

This project implements a simple **Executor Agent** that executes plans in a simulated Android environment (`MockAndroidEnv`). The agent can dynamically **replan** if a step fails, using a basic `PlannerAgent`. A `VerifierAgent` checks if the goals were achieved successfully.

---

## Features

- Simulated Android environment with basic Wi-Fi toggling actions
- Planner that generates plans based on goal and current environment state
- Executor that executes plans step-by-step, triggers replanning on failures
- Verifier that confirms if goals were achieved based on UI state
- Prevention of infinite replanning loops

---

## Files

- `executor_agent.py` — main code including environment, planner, executor, verifier, and example usage

---

## Requirements

- Python 3.7 or higher
- No external libraries required (uses only built-in `json`)

---

## How to Run

1. Clone or download this repository.

2. Open a terminal and navigate to the directory containing `executor_agent.py`.

3. Run the script:

```bash
python executor_agent.py
