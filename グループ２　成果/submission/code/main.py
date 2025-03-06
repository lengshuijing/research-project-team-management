import json
import random
from difflib import SequenceMatcher
from pydantic import BaseModel
from openai import OpenAI
import textgrad as tg
import concurrent.futures

client = OpenAI()
tg.set_backward_engine("gpt-4o-mini")


class ResearchFinding(BaseModel):
    finding: str


def load_hard_problem(json_path="./hard_problems.json"):
    with open(json_path, "r") as f:
        data = json.load(f)
    topic = random.choice(data["topics"])
    return random.choice(topic["openQuestions"])


def vanilla_scientist(hard_problem):
    print("Querying for initial research finding...")
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a novelty-driven knowledge aggregator. Return a single, concise research finding that is a logical and innovative scientific fact addressing both the mental block and the key knowledge gap.",
            },
            {
                "role": "user",
                "content": f"What is your first mental block when trying to solve {hard_problem}? What do you wish you knew completely about solving this problem?",
            },
        ],
        response_format=ResearchFinding,
    )
    result = completion.choices[0].message.parsed.finding
    print("Initial research finding:", result)
    return result


def wish_i_knew(vanilla_scientist_text):
    print("Extracting 'wish I knew'...")
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a novelty-driven knowledge aggregator. Return a single, concise research fact that encapsulates the key knowledge gap as a novel scientific intuition.",
            },
            {
                "role": "user",
                "content": f"Extract and fabricate a concise scientific fact that states what you wish you knew completely: {vanilla_scientist_text}",
            },
        ],
        response_format=ResearchFinding,
    )
    result = completion.choices[0].message.parsed.finding
    print("'Wish I knew':", result)
    return result


def first_mental_block(vanilla_scientist_text):
    print("Extracting first mental block...")
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a novelty-driven knowledge aggregator. Return a single, concise research fact that logically states the primary mental block.",
            },
            {
                "role": "user",
                "content": f"Extract the primary mental block from: {vanilla_scientist_text}",
            },
        ],
        response_format=ResearchFinding,
    )
    result = completion.choices[0].message.parsed.finding
    print("First mental block:", result)
    return result


def attempt_1(wish_i_knew_text, first_mental_block_text, given=None):
    if given:
        system_prompt = f"You are a novelty-driven knowledge aggregator that transforms creative intuition into scientific facts. Given the fact: {given}. Your 'wish I knew' is: {wish_i_knew_text}. Develop a novel, logically sound research finding that overcomes the mental block below."
    else:
        system_prompt = f"You are a novelty-driven knowledge aggregator that transforms creative intuition into scientific facts. Your 'wish I knew' is: {wish_i_knew_text}. Develop a novel, logically sound research finding that addresses the following mental block."
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Solve this mental block: {first_mental_block_text} as a single, concise scientific fact.",
            },
        ],
        response_format=ResearchFinding,
    )
    return completion.choices[0].message.parsed.finding


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def recursive_attempt(
    wish_i_knew_text, first_mental_block_text, max_steps=5, convergence_threshold=0.9
):
    print("Recursively refining outcome...")
    previous_outcome = None
    current_outcome = None
    for step in range(1, max_steps + 1):
        print(f"Recursion step {step}...")
        current_outcome = attempt_1(
            wish_i_knew_text, first_mental_block_text, given=previous_outcome
        )
        print("Current outcome:", current_outcome)
        if previous_outcome:
            sim = similarity(current_outcome, previous_outcome)
            print("Similarity to previous:", sim)
            if sim >= convergence_threshold:
                print("Convergence reached.")
                break
        previous_outcome = current_outcome
    return current_outcome


def ask_why(fact):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a debate facilitator for scientific inquiry. Provide a concise, logical, and novel explanation that answers 'Why is this research finding true?' as a single research fact.",
            },
            {
                "role": "user",
                "content": f"Given the research finding '{fact}', why is this fact true? Provide a single, concise research fact.",
            },
        ],
        response_format=ResearchFinding,
    )
    return completion.choices[0].message.parsed.finding


def multi_agent_debate(fact, num_agents=5):
    print("Launching multi-agent debate...")
    findings = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_agents) as executor:
        futures = [executor.submit(ask_why, fact) for _ in range(num_agents)]
        for future in concurrent.futures.as_completed(futures):
            try:
                findings.append(future.result())
            except:
                findings.append("Error")
    return findings


def refine_with_textgrad(outcome):
    print("Refining with TextGrad...")
    solution = tg.Variable(
        outcome,
        requires_grad=True,
        role_description="research finding (a concise scientific fact)",
    )
    loss_fn = tg.TextLoss(
        "Evaluate for novelty, clarity, scientific rigor. Refine into a concise scientific fact."
    )
    optimizer = tg.TGD(parameters=[solution])
    loss = loss_fn(solution)
    loss.backward()
    optimizer.step()
    result = solution.value
    print("Refined outcome:", result)
    return result


class BaseAgent:
    def __init__(self, agent_id, rank):
        self.agent_id = agent_id
        self.rank = rank


class SoldierAgent(BaseAgent):
    def execute_task(self, problem):
        print(f"{self.rank} {self.agent_id} executing task on:", problem)
        vs = vanilla_scientist(problem)
        w = wish_i_knew(vs)
        b = first_mental_block(vs)
        final_outcome = recursive_attempt(w, b, 5, 0.9)
        print(f"{self.rank} {self.agent_id} outcome:", final_outcome)
        return final_outcome


class OfficerAgent(BaseAgent):
    def process_result(self, result):
        print(f"{self.rank} {self.agent_id} processing result...")
        return refine_with_textgrad(result)


class GeneralAgent(BaseAgent):
    def full_command_chain(self, problem):
        print(f"{self.rank} {self.agent_id} initiating command chain...")
        soldier = SoldierAgent("S1", "Soldier")
        outcome = soldier.execute_task(problem)
        officer = OfficerAgent("O1", "Officer")
        refined = officer.process_result(outcome)
        print(f"{self.rank} {self.agent_id} final refined result:", refined)
        return refined


def main():
    print("Loading a hard problem...")
    hard_problem = load_hard_problem()
    print("Selected problem:", hard_problem)
    general = GeneralAgent("G1", "General")
    final_outcome = general.full_command_chain(hard_problem)
    print("Saving final outcome...")
    with open("final_research_finding.txt", "w") as f:
        f.write("Final Research Finding:\n")
        f.write(final_outcome + "\n")
    print("Debate on final outcome...")
    debate_results = multi_agent_debate(final_outcome, 5)
    with open("final_research_finding.txt", "a") as f:
        f.write("\nMulti-Agent Debate Findings (Why?):\n")
        for i, finding in enumerate(debate_results, start=1):
            f.write(f"{i}. {finding}\n")
    print("Done.")


if __name__ == "__main__":
    main()
