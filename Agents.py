import os
import time
import pandas as pd
from tqdm import tqdm
from autogen_core.models import ModelInfo, ModelFamily
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
import json
import re

# Load types from types.json
with open(os.path.join(os.path.dirname(__file__), "types.json"), "r", encoding="utf-8") as f:
    types_data = json.load(f)
ISSUE_TYPES = types_data["issue_types"]
BIAS_TYPES = types_data["bias_types"]

# Model info setup
llama4_maverick_info = ModelInfo(vision=False, function_calling=True, json_output=True, structured_output=False, family=ModelFamily.LLAMA_4_MAVERICK)
#my_model_info = ModelInfo(vision=True, function_calling=True, json_output=True, structured_output=True, family=ModelFamily.ANY)

# Agent pools
AGENT_POOLS = {
	"Critic": [
		{"name": "Critic_Groq", "model": "meta-llama/llama-4-maverick-17b-128e-instruct", "api_key": os.getenv("GROQ_API_KEY_1"), "base_url": "https://api.groq.com/openai/v1", "model_info": llama4_maverick_info},
		{"name": "Critic_Groq", "model": "meta-llama/llama-4-maverick-17b-128e-instruct", "api_key": os.getenv("GROQ_API_KEY_2"), "base_url": "https://api.groq.com/openai/v1", "model_info": llama4_maverick_info},
		{"name": "Critic_Groq", "model": "meta-llama/llama-4-maverick-17b-128e-instruct", "api_key": os.getenv("GROQ_API_KEY_3"), "base_url": "https://api.groq.com/openai/v1", "model_info": llama4_maverick_info},
	],
	"Checker": [
		{"name": "Checker_Groq", "model": "meta-llama/llama-4-maverick-17b-128e-instruct", "api_key": os.getenv("GROQ_API_KEY_1"), "base_url": "https://api.groq.com/openai/v1", "model_info": llama4_maverick_info},
		{"name": "Checker_Groq", "model": "meta-llama/llama-4-maverick-17b-128e-instruct", "api_key": os.getenv("GROQ_API_KEY_2"), "base_url": "https://api.groq.com/openai/v1", "model_info": llama4_maverick_info},
		{"name": "Checker_Groq", "model": "meta-llama/llama-4-maverick-17b-128e-instruct", "api_key": os.getenv("GROQ_API_KEY_3"), "base_url": "https://api.groq.com/openai/v1", "model_info": llama4_maverick_info},
	],
	"ReEval": [
		{"name": "ReEval_Groq", "model": "meta-llama/llama-4-maverick-17b-128e-instruct", "api_key": os.getenv("GROQ_API_KEY_1"), "base_url": "https://api.groq.com/openai/v1", "model_info": llama4_maverick_info},
		{"name": "ReEval_Groq", "model": "meta-llama/llama-4-maverick-17b-128e-instruct", "api_key": os.getenv("GROQ_API_KEY_2"), "base_url": "https://api.groq.com/openai/v1", "model_info": llama4_maverick_info},
		{"name": "ReEval_Groq", "model": "meta-llama/llama-4-maverick-17b-128e-instruct", "api_key": os.getenv("GROQ_API_KEY_3"), "base_url": "https://api.groq.com/openai/v1", "model_info": llama4_maverick_info},
	],
	"MetricAgent": [
		{"name": "MetricAgent_Groq", "model": "meta-llama/llama-4-maverick-17b-128e-instruct", "api_key": os.getenv("GROQ_API_KEY_1"), "base_url": "https://api.groq.com/openai/v1", "model_info": llama4_maverick_info}
	]
}

AGENT_COOLDOWN_SECONDS = 60

def init_agent_status_and_idx():
	agent_status = {config["name"]: {"cooldown_until": 0} for pool in AGENT_POOLS.values() for config in pool}
	next_agent_idx = {role: 0 for role in AGENT_POOLS}
	return agent_status, next_agent_idx

def select_agent_config(role: str, agent_status, next_agent_idx) -> dict:
	pool = AGENT_POOLS[role]
	start_idx = next_agent_idx[role]
	for i in range(len(pool)):
		idx = (start_idx + i) % len(pool)
		config = pool[idx]
		agent_name = config["name"]
		if time.time() >= agent_status[agent_name]["cooldown_until"]:
			next_agent_idx[role] = (idx + 1) % len(pool)
			return config
	raise RuntimeError(f"All agents for role '{role}' are on cooldown.")

def get_client_from_config(config: dict) -> OpenAIChatCompletionClient:
	return OpenAIChatCompletionClient(
		model=config["model"],
		api_key=config["api_key"],
		base_url=config["base_url"],
		model_info=config["model_info"],
		include_name_in_message=False
	)

def place_agent_on_cooldown(agent_name: str, agent_status):
	cooldown_time = time.time() + AGENT_COOLDOWN_SECONDS
	agent_status[agent_name]["cooldown_until"] = cooldown_time

async def process_csv(INPUT_CSV, GOOD_CSV, ISSUES_CSV, LOG_FILE, columns_to_check=None, issue_types=None, bias_types=None):
	df = pd.read_csv(INPUT_CSV)
	agent_status, next_agent_idx = init_agent_status_and_idx()
	if os.path.exists(GOOD_CSV):
		os.remove(GOOD_CSV)
	if os.path.exists(ISSUES_CSV):
		os.remove(ISSUES_CSV)
	pd.DataFrame(columns=df.columns).to_csv(GOOD_CSV, index=False)
	pd.DataFrame(columns=list(df.columns) + ["issues", "bias"]).to_csv(ISSUES_CSV, index=False)
	with open(LOG_FILE, "w", encoding="utf-8") as log:
		log.write("Processing started...\n")
	# Use provided types or fallback to static ones
	if issue_types is None:
		issue_types = ISSUE_TYPES
	if bias_types is None:
		bias_types = BIAS_TYPES
	for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
		# Build a dynamic prompt for selected columns
		if columns_to_check is None:
			columns_to_check = list(df.columns)
		col_data = "\n".join([f"{col}: {row[col]}" for col in columns_to_check])
		prompt = (
			f"Analyze the following data for any possible issues and bias.\n"
			f"{col_data}\n\n"
			f"Choose ONLY from these issue types: {issue_types}.\n"
			f"Choose ONLY from these bias types: {bias_types}.\n"
			"Respond with a JSON object with two fields: 'issues' (list of selected issue types) and 'bias' (list of selected bias types).\n"
			"If no issue or bias is present, respond with ['none'] for both fields. Do not invent new types. Example:\n"
			'{"issues": ["none"], "bias": ["none"]}'
		)
		active_team_configs = {}
		while True:
			try:
				critic_config = select_agent_config("Critic", agent_status, next_agent_idx)
				checker_config = select_agent_config("Checker", agent_status, next_agent_idx)
				reeval_config = select_agent_config("ReEval", agent_status, next_agent_idx)
				active_team_configs = {
					"Critic_Agent": critic_config,
					"Checker_Agent": checker_config,
					"Re_Eval_Agent": reeval_config,
				}
				critic_agent = AssistantAgent(
					name="Critic_Agent",
					model_client=get_client_from_config(critic_config),
					system_message="You are a data quality and bias detection expert. For each row, analyze for any possible issues and bias. Respond with a JSON object: { 'issues': [...], 'bias': [...] }"
				)
				checker_agent = AssistantAgent(
					name="Checker_Agent",
					model_client=get_client_from_config(checker_config),
					system_message="You are a data quality and bias detection expert. For each row, analyze for any possible issues and bias. Respond with a JSON object: { 'issues': [...], 'bias': [...] }"
				)
				re_eval_agent = AssistantAgent(
					name="Re_Eval_Agent",
					model_client=get_client_from_config(reeval_config),
					system_message="You are a data quality and bias detection expert. For each row, analyze for any possible issues and bias. Respond with a JSON object: { 'issues': [...], 'bias': [...] }"
				)
				team = RoundRobinGroupChat(
					participants=[critic_agent, checker_agent, re_eval_agent],
					max_turns=3
				)
				task = TextMessage(content=prompt.strip(), source="user")
				start_agent = time.time()
				result = await team.run(task=task)
				end_agent = time.time()
				last_msg = result.messages[-1].content.strip()
				# Try to parse the agent's response as JSON
				try:
					# Extract JSON from response, remove backticks and extra text
					cleaned = last_msg.strip()
					# Remove triple backticks and language tags
					cleaned = re.sub(r'^```json|^```|```$', '', cleaned, flags=re.MULTILINE).strip()
					# Find first JSON object in the string
					match = re.search(r'\{.*\}', cleaned, re.DOTALL)
					if match:
						cleaned_json = match.group(0)
						parsed = json.loads(cleaned_json)
						issues = parsed.get("issues", ["none"])
						bias = parsed.get("bias", ["none"])
						if (all(i in ISSUE_TYPES for i in issues) and all(b in BIAS_TYPES for b in bias)):
							print(f"\nRow {idx} → Issues: {issues}, Bias: {bias}")
							if issues == ["none"] and bias == ["none"]:
								pd.DataFrame([row]).to_csv(GOOD_CSV, mode="a", header=False, index=False)
							else:
								row_copy = row.copy()
								row_copy["issues"] = ", ".join(issues)
								row_copy["bias"] = ", ".join(bias)
								pd.DataFrame([row_copy]).to_csv(ISSUES_CSV, mode="a", header=False, index=False)
							with open(LOG_FILE, "a", encoding="utf-8") as log:
								log.write(f"Row {idx}: Issues: {issues}, Bias: {bias}\n")
						else:
							raise ValueError("Response contains types not in allowed lists.")
					else:
						raise ValueError("No JSON object found in response.")
				except Exception:
					issues = ["parse_error"]
					bias = ["parse_error"]
					# Fallback: show raw agent response for debugging
					row_copy = row.copy()
					row_copy["issues"] = "parse_error"
					row_copy["bias"] = "parse_error"
					row_copy["raw_response"] = last_msg
					pd.DataFrame([row_copy]).to_csv(ISSUES_CSV, mode="a", header=False, index=False)
					with open(LOG_FILE, "a", encoding="utf-8") as log:
						log.write(f"Row {idx}: PARSE ERROR. Raw response: {last_msg}\n")
					print(f"\nRow {idx} → PARSE ERROR. Raw response: {last_msg}")
				break
			except Exception as e:
				err_msg = str(e)
				print(f"\nERROR on row {idx}: {err_msg}")
				if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
					failing_agent_identified = False
					for agent_name_in_team, config in active_team_configs.items():
						if agent_name_in_team in err_msg or config['model'] in err_msg:
							place_agent_on_cooldown(config["name"], agent_status)
							failing_agent_identified = True
							break
					if not failing_agent_identified:
						print("⚠ Could not identify the failing agent. Cooling down ALL agents from this attempt.")
						for config in active_team_configs.values():
							place_agent_on_cooldown(config["name"], agent_status)
					print("Retrying row with the next available agents...")
					time.sleep(1)
					continue
				else:
					with open(LOG_FILE, "a", encoding="utf-8") as log:
						log.write(f"Row {idx} SKIPPED due to UNHANDLED ERROR: {err_msg}\n")
					break
