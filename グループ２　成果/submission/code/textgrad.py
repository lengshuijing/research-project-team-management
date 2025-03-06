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


def load_hard_problem(json_path: str = "./hard_problems.json") -> str:
    with open(json_path, "r") as f:
        data = json.load(f)
    topic = random.choice(data["topics"])
    hard_problem = random.choice(topic["openQuestions"])
    return hard_problem


def vanilla_scientist(hard_problem: str) -> str:
    prompt_text = (
        f"What is your first mental block when trying to solve {hard_problem}? "
        "What do you wish you knew completely about solving this problem?"
    )
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a novelty-driven knowledge aggregator. Return a single, concise research finding "
                    "that is a logical and innovative scientific fact addressing both the mental block and the key knowledge gap."
                ),
            },
            {"role": "user", "content": prompt_text},
        ],
        response_format=ResearchFinding,
    )
    return completion.choices[0].message.parsed.finding


def wish_i_knew(vanilla_scientist_text: str) -> str:
    prompt = (
        "Extract and fabricate a concise scientific fact from the following text that states what you wish you knew completely, "
        "thereby capturing the key knowledge gap (your intuition): "
        f"{vanilla_scientist_text}"
    )
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a novelty-driven knowledge aggregator. Return a single, concise research fact "
                    "that encapsulates the key knowledge gap as a novel scientific intuition."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format=ResearchFinding,
    )
    return completion.choices[0].message.parsed.finding


def first_mental_block(vanilla_scientist_text: str) -> str:
    prompt = (
        "Extract and express as a concise scientific fact the primary mental block in the following text: "
        f"{vanilla_scientist_text}"
    )
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a novelty-driven knowledge aggregator. Return a single, concise research fact "
                    "that logically states the primary mental block."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format=ResearchFinding,
    )
    return completion.choices[0].message.parsed.finding


def attempt_1(
    wish_i_knew_text: str, first_mental_block_text: str, given: str = None
) -> str:
    if given:
        system_prompt = (
            "You are a novelty-driven knowledge aggregator that transforms creative intuition into scientific facts. "
            f"Given the fact: {given}. Your 'wish I knew' (intuition) is: {wish_i_knew_text}. "
            "Develop a novel, logically sound research finding that overcomes the mental block below."
        )
    else:
        system_prompt = (
            "You are a novelty-driven knowledge aggregator that transforms creative intuition into scientific facts. "
            f"Your 'wish I knew' (intuition) is: {wish_i_knew_text}. "
            "Develop a novel, logically sound research finding that addresses the following mental block."
        )
    user_prompt = (
        "Solve this mental block: "
        f"{first_mental_block_text} "
        "and express your solution as a single, concise scientific fact."
    )
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=ResearchFinding,
    )
    return completion.choices[0].message.parsed.finding


def similarity(a: str, b: str) -> float:
    from difflib import SequenceMatcher

    return SequenceMatcher(None, a, b).ratio()


def recursive_attempt(
    wish_i_knew_text: str,
    first_mental_block_text: str,
    max_steps: int = 5,
    convergence_threshold: float = 0.9,
) -> str:
    previous_outcome = None
    current_outcome = None

    for step in range(1, max_steps + 1):
        print(f"\n--- Recursion Step {step} ---")
        current_outcome = attempt_1(
            wish_i_knew_text=wish_i_knew_text,
            first_mental_block_text=first_mental_block_text,
            given=previous_outcome,
        )
        print(f"Research finding at step {step}: {current_outcome}")

        if previous_outcome is not None:
            sim = similarity(current_outcome, previous_outcome)
            print(f"Similarity with previous outcome: {sim:.3f}")
            if sim >= convergence_threshold:
                print("Meta-agent: Convergence detected. Halting recursion.")
                break

        previous_outcome = current_outcome

    return current_outcome


def ask_why(fact: str) -> str:
    prompt_text = (
        f"Given the research finding: '{fact}', why is this fact true? "
        "Provide a concise, logical explanation as a single research finding."
    )
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a debate facilitator for scientific inquiry. Your task is to provide a concise, logical, and novel explanation "
                    "that answers 'Why is this research finding true?' Ensure your answer is a single, well-founded research fact."
                ),
            },
            {"role": "user", "content": prompt_text},
        ],
        response_format=ResearchFinding,
    )
    return completion.choices[0].message.parsed.finding


def multi_agent_debate(fact: str, num_agents: int = 5) -> list:
    findings = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_agents) as executor:
        futures = [executor.submit(ask_why, fact) for _ in range(num_agents)]
        for future in concurrent.futures.as_completed(futures):
            try:
                findings.append(future.result())
            except Exception as e:
                findings.append(f"Error: {e}")
    return findings


def refine_with_textgrad(outcome: str) -> str:
    solution = tg.Variable(
        outcome,
        requires_grad=True,
        role_description="research finding (a concise scientific fact)",
    )
    loss_fn = tg.TextLoss(
        "Evaluate the following research finding for its novelty, clarity, and scientific rigor. "
        "Identify any lack of novelty, redundancy, or logical weakness, and refine the statement into a concise, "
        "scientifically sound fact that resolves the mental block. Return only a single, refined research fact."
    )
    optimizer = tg.TGD(parameters=[solution])
    loss = loss_fn(solution)
    loss.backward()
    optimizer.step()
    return solution.value


def main():
    hard_problem = load_hard_problem("./hard_problems.json")
    print("=== Selected Hard Problem ===")
    print(hard_problem, "\n")

    vanilla_text = vanilla_scientist(hard_problem)
    print("=== Initial Research Finding (vanilla_scientist) ===")
    print(vanilla_text, "\n")

    wish_i_knew_text = wish_i_knew(vanilla_text)
    print("=== Extracted 'Wish I Knew' (Intuition as a Scientific Fact) ===")
    print(wish_i_knew_text, "\n")

    first_mental_block_text = first_mental_block(vanilla_text)
    print("=== Extracted 'First Mental Block' (Research Finding) ===")
    print(first_mental_block_text, "\n")

    print("=== Recursively Generating New Findings ===")
    final_outcome = recursive_attempt(
        wish_i_knew_text=wish_i_knew_text,
        first_mental_block_text=first_mental_block_text,
        max_steps=5,
        convergence_threshold=0.9,
    )

    refined_outcome = refine_with_textgrad(final_outcome)

    output_file = "final_research_finding.txt"
    with open(output_file, "w") as f:
        f.write("Final Research Finding:\n")
        f.write(refined_outcome + "\n")
    print("\n=== Final Refined Research Finding Saved to", output_file, "===\n")
    print(refined_outcome, "\n")

    print("=== Multi-Agent Debate: Asking 'Why?' ===")
    debate_findings = multi_agent_debate(refined_outcome, num_agents=5)
    with open(output_file, "a") as f:
        f.write("\nMulti-Agent Debate Findings (Why?):\n")
        for idx, finding in enumerate(debate_findings, start=1):
            debate_line = f"{idx}. {finding}\n"
            f.write(debate_line)
            print(debate_line.strip())

    print("\nAll findings have been appended to", output_file)


if __name__ == "__main__":
    main()
