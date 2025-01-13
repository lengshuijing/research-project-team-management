#Required Libraries
from pathlib import Path
from autogen import AssistantAgent
import pandas as pd
import numpy as np
import json

# Configuration for Ollama
OLLAMA_CONFIG = {
    "model": "llama3.2:latest",  # Specify the model
    "api_type": "ollama",       # Use Ollama API
    "client_host": "127.0.0.1:11434"  # Default Ollama endpoint
}

# Global Variables
MAX_NODE_NUMBER = 10  # Maximum number of nodes (agents) in the swarm
MAX_TOKEN = 500  # Max tokens for generation
MODEL_NAME = "llama3.2:latest"  # Model name
API_TYPE = "ollama"  # API type

# Cases
# 1: initial round
CASE_INITIAL_ROUND = 1
# 2: debate again with same topology
CASE_ADDITIONAL_DEBATE = 2
# 3: rearrange specializations and debate
CASE_REARRANGE = 3
# 4: when a conclusion is reached
CASE_ENDED = 4

# Initialize Assistant Agent
assistant_agent = AssistantAgent(
    name="DNAAssistant",
    system_message="You are a helpful assistant for resolving tasks with a focus on collaboration and problem-solving.",
    llm_config={
        "config_list": [
            {
                "model": OLLAMA_CONFIG["model"],
                "api_type": OLLAMA_CONFIG["api_type"],
                "client_host": OLLAMA_CONFIG["client_host"]
            }
        ]
    }
)

# Functions
def specialists_divider(response):
    """
    Divide the roles for specialists based on the response.

    Parameters:
    - response: The plain-text response from the DNA node.

    Returns:
    - specialists_group: A dictionary mapping roles to the number of specialists.
    - total_number: The total number of specialists required.
    """
    # Ensure the response is valid JSON or plain text
    try:
        # Attempt to parse the response as JSON
        parsed_response = json.loads(response)
        specialists = parsed_response['role of the specialists']
        numbers = parsed_response['number of each specialists']

        specialists_group = {}
        for i, specialist in enumerate(specialists):
            specialists_group[specialist] = numbers[i]

        total_number = parsed_response['number of specialists']
    except json.JSONDecodeError:
        # Handle plain text by parsing manually (example below)
        print("Warning: Response is not JSON, attempting manual parsing...")
        specialists_group = {}
        total_number = 0
        for line in response.splitlines():
            if "specialist" in line.lower():
                # Example of extracting specialist information
                role = line.split(":")[0].strip()
                count = 1  # Default count for plain text
                specialists_group[role] = count
                total_number += count

    return specialists_group, total_number

def case_parser(final_response):
    """
    Determine the current case to proceed with based on the DNA node's answer.

    Parameters:
    - final_response: The final response from the DNA node.

    Returns:
    - The corresponding case constant.
    """
    # Parse the final response
    parsed_response = eval(final_response)
    if parsed_response['satisfactory'] == 'Y':
        return CASE_ENDED
    elif parsed_response['debate/rearrange'] == 'D':
        return CASE_ADDITIONAL_DEBATE
    elif parsed_response['debate/rearrange'] == 'R':
        return CASE_REARRANGE

# Example of interaction with the AssistantAgent
def run_example():
    """
    Run an example interaction with the AssistantAgent.
    """
    prompt = "Please classify the specialists required to solve a math problem."
    messages = [
        {"role": "user", "content": prompt}
    ]

    # Query the AssistantAgent
    response = assistant_agent.generate_reply(messages=messages)

    # Parse the response for specialists
    print("Response:", response)
    specialists_group, total_number = specialists_divider(response["content"])
    print("Specialists Group:", specialists_group)
    print("Total Specialists:", total_number)

# Test the example
run_example()

def extract_response(final_response):
    """
    Extracts the numeric response from the final answer of the DNA node.

    Parameters:
    - final_response: The full response text.

    Returns:
    - The numeric portion of the response.
    """
    messages = [
        {
            "role": "user",
            "content": f'Extract the numeric answer from this text: {final_response}',
        },
        {
            "role": "system",
            "content": "Only reply with numbers (the numeric response)."
        }
    ]
    response = assistant_agent.generate_reply(messages=messages)
    return response.get("content", "No response content")


# Class Definition
class DnaNode:
    def __init__(self, problem):
        """
        Initialize a DNA Node.

        Parameters:
        - problem: The problem statement to solve.
        """
        self.problem = problem
        self.specialists = {}
        self.stem_cell_nodes = {}

    def run(self, answer_format, query, temperature=0.5):
        """
        Request an answer to the query in a specified format.

        Parameters:
        - answer_format: The format of the expected response.
        - query: The question or task to be answered.
        - temperature: The temperature for response randomness.

        Returns:
        - The formatted response as a dictionary.
        """
        messages = [
            {
                "role": "user",
                "content": f'{query}, please answer the question in the following format: "{answer_format}". Do not give anything except a Python dictionary. Answer within a maximum token limit of {MAX_TOKEN}.',
            }
        ]
        response = assistant_agent.generate_reply(messages=messages)
        return response.get("content", "No response content")

    def assign_specialists(self):
        messages = [
            {
                "role": "user",
                "content": f'''
                    You are assigned a task as follows: "{self.problem}". Please decide the number of specialists
                    and specify their roles in JSON format.
                    Example format:
                    {{
                        "number of specialists": 6,
                        "role of the specialists": ["mathematician", "physician"],
                        "number of each specialists": [2, 4]
                    }}
                '''
            }
        ]
        response = assistant_agent.generate_reply(messages=messages)
        specialists = response.get("content", "No response content")
        self.specialists = specialists_divider(specialists)[0]

        specialist_nodes = {}
        for specialist in self.specialists:
            stem_cell_node = StemCellNode(problem=self.problem, specialization=specialist)
            specialist_nodes[specialist] = stem_cell_node

        self.stem_cell_nodes = specialist_nodes
        return specialists


    def reassign_specialists(self):
        """
        Reassign specialists and modify the group based on the previous structure.

        Returns:
        - specialists: A JSON object containing the reassigned specialist group information.
        """
        prompt = f'''You are assigned a particular task as follows: "{self.problem}", and you can use up to {MAX_NODE_NUMBER} specialists to solve this task.
        Please decide the number of specialists you want to use and specify their roles (e.g., mathematician, physician, psychologist, etc.). Use more specialists in one particular field if needed.
        Please also balance cost and efficiency. You previously used the following specialist distribution and decided to change it: "{self.specialists}".'''

        answer_format = {
            "number of specialists": 6,
            "role of the specialists": ["mathematician", "physician"],
            "number of each specialists": [2, 4],
            "reason": "Explain why you want to use this many specialists and divide them this way.",
            "question": "Repeat the query."
        }

        specialists = self.run(answer_format=answer_format, query=prompt)
        self.specialists = specialists_divider(specialists)[0]

        specialist_nodes = {}
        for specialist in self.specialists:
            stem_cell_node = StemCellNode(problem=self.problem, specialization=specialist)
            specialist_nodes[specialist] = stem_cell_node

        self.stem_cell_nodes = specialist_nodes
        return specialists

    def combine_answers(self, concentrated_response):
        """
        Combine answers from specialists into one unified response.

        Parameters:
        - concentrated_response: The responses from the specialist group.

        Returns:
        - A combined and balanced response.
        """
        prompt = f'''
        You previously designated a specialist group as follows: "{self.specialists}" to solve this question: "{self.problem}". Here are the answers from the specialist group: "{concentrated_response}".
        Combine all the answers into one response. Try to balance details and length. Provide your response within {MAX_TOKEN} tokens.'''

        answer_format = {
            "combined answer": "Write your combined answer here.",
            "satisfactory": "Y/N, answer Y if satisfied. Answer N otherwise.",
            "debate/rearrange": "D/R/NA. Answer D to debate again, R to rearrange, NA if no further action is needed."
        }

        response = self.run(query=prompt, answer_format=answer_format)
        return response

    def no_debate_round(self, answer_format):
        """
        Simulate a no-debate round where each specialist provides an answer.

        Parameters:
        - answer_format: Expected format of the response.

        Returns:
        - A dictionary of responses from each specialist.
        """
        responses = {}
        for specialist, count in self.specialists.items():
            responses[specialist] = []
            for _ in range(count):
                prompt = f"As a {specialist}, please answer the following: {self.problem}"
                response = assistant_agent.generate_reply(messages=[
                    {"role": "user", "content": prompt}
                ])
                responses[specialist].append(response.get("content", "No response content"))
        return responses

class StemCellNode:
    """
    A specialist node. Contains the query of the problem to resolve and the specific specialization.
    """

    def __init__(self, problem, specialization):
        self.problem = problem
        self.specialization = specialization

    def run(self, prompt, answer_format, temperature=0.5):
        """
        Request an answer to the query in a specified format.

        Parameters:
        - answer_format: The format in which you'd like the response to be structured.
        - max_token: The maximum number of tokens in the response to control response length.

        Returns:
        - response: a response to the query, formatted as specified by `answer_format` and within the `max_token` limit.
        """
        messages = [
            {
                "role": "user",
                "content": f'{prompt}, please answer the question in the following format: "{answer_format}", do not give anything except a JSON object. Answer within maxium token of {MAX_TOKEN}.',
            },
            {
                "role": "system",
                "content": f'You are a {self.specialization}.'
            }
        ]

        response = assistant_agent.chat(messages=messages)
        return response.content

    def answer_combiner(self, answer, answer_format):
        """ Combines a set of answers into one larger text."""
        query = f'''
        Combine the following answer into one answer: {answer}. Try to balance details and length. Give your answer within {MAX_TOKEN} tokens.
        '''
        response = self.run(prompt=query, answer_format=answer_format)
        return response

# Combine responses
def combine_responses(response, dna_node):
    """
    Combines and concentrates a set of answers into a shorter and summarized response.

    Parameters:
    - response: Dictionary of responses from specialists.
    - dna_node: The DNA node instance.

    Returns:
    - concentrated_response: Final answers per specialist.
    - combined_response: Aggregated responses per specialist.
    - new_response: Updated responses from specialists.
    """
    concentrated_response = {}
    new_response = {}
    combined_response = {}

    for specialist in response:
        if len(response[specialist]) == 1:
            concentrated_response[specialist] = response[specialist]
        else:
            new_response[specialist] = []
            combined_response[specialist] = []

            for i in range(len(response[specialist])):
                query = f'''
                    The problem you are solving is: "{dna_node.problem}". Here are the responses from other {specialist}s in your group: "{response[specialist][i]}". Here is your original response: "{response[specialist][:i] + response[specialist][i+1:]}". Do you think you can improve your response?
                '''
                answer_format = {
                    "improvement": "Y/N, answer Y if you think your original response needs improvement. Answer N otherwise",
                    "response": "Update your response",
                    "reason": "Where have you improved? What have you learnt from the other specialists?",
                    "further_improvement": "Y/N, answer Y if you think your updated response needs further improvement. Answer N otherwise"
                }
                ans = dna_node.stem_cell_nodes[specialist].run(prompt=query, answer_format=answer_format)
                new_response[specialist].append(ans)
                combined_response[specialist].append(ans)

            total = len(combined_response[specialist])
            no_further_improvement = sum(
                1 for item in combined_response[specialist]
                if eval(item).get('further_improvement', 'N') == 'N'
            )
            if no_further_improvement / total >= 0.5:
                print(f'{specialist} arrived at the final conclusion')
                concentrated_response[specialist] = dna_node.combine_answers(combined_response[specialist])

    return concentrated_response, combined_response, new_response


# Execution Loop
# Execution Loop
current_round = CASE_INITIAL_ROUND

# Define a query and specify any conditions for the answer format.
query = "Toula went to the bakery and bought various types of pastries. She bought 3 dozen donuts which cost $68 per dozen, 2 dozen mini cupcakes which cost $80 per dozen, and 6 dozen mini cheesecakes for $55 per dozen. How much was the total cost?"
# Correct answer: 694 for testing
answer_format = {
    "response": "Write your response here."
}

# Initialize DNA Node to split the agents.
dna_node = DnaNode(query)

while current_round != CASE_ENDED:
    if current_round == CASE_INITIAL_ROUND:
        # Initial round, no specialists assigned
        dna_node.assign_specialists()
        print(f"Specialists assigned: {dna_node.specialists}")

        # Each agent gives an initial answer
        response = dna_node.no_debate_round(answer_format=answer_format)
        print(f"Initial responses: {response}")

        # Within group debate
        concentrated_response, combined_response, new_response = combine_responses(response, dna_node)

        # DNA Node makes its final decision
        final_response = dna_node.combine_answers(concentrated_response)
        print("FINAL RESPONSE")
        print(final_response)

        current_round = case_parser(final_response)

    elif current_round == CASE_ADDITIONAL_DEBATE:
        # Keep specialist distribution, debate again.
        response = dna_node.debate_round(answer_format=answer_format)

        # Within group debate
        concentrated_response, combined_response, new_response = combine_responses(response, dna_node)

        # DNA Node makes its final decision
        final_response = dna_node.combine_answers(concentrated_response)
        print("FINAL RESPONSE")
        print(final_response)

        current_round = case_parser(final_response)

    elif current_round == CASE_REARRANGE:
        # Rearrange specialists and have a new debate round
        dna_node.reassign_specialists()
        print(f"Reassigned specialists: {dna_node.specialists}")

        # Each agent gives an initial answer
        response = dna_node.no_debate_round(answer_format=answer_format)

        # Within group debate
        concentrated_response, combined_response, new_response = combine_responses(response, dna_node)

        # DNA Node makes its final decision
        final_response = dna_node.combine_answers(concentrated_response)
        print("FINAL RESPONSE")
        print(final_response)

        current_round = case_parser(final_response)

# Get final response
parsed_final_response = extract_response(final_response)
print(f"Extracted final response: {parsed_final_response}")


