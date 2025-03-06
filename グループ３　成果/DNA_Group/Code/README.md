
# Multi-Agent Discussion System with Organizer

This project is a Python-based multi-agent discussion system where an intelligent organizer coordinates multiple specialized agents to collaboratively solve complex questions. The organizer determines question difficulty, identifies relevant areas of study, assigns roles to agents, and checks the quality of responses iteratively until a satisfactory answer is achieved. The system also supports follow-up questions, allowing for deeper exploration of topics.

## Features
- **Intelligent Organizer**: Assesses question difficulty, identifies relevant areas of study, assigns agents, and checks satisfaction.
- **Specialized Agents**: Agents provide insights from specific fields like mathematics, physics, biology, etc.
- **Iterative Feedback**: Organizer repeats discussions with agents until a satisfactory answer is reached.
- **Follow-up Question Support**: Users can provide follow-up questions for additional discussions.

## Requirements

### System Requirements
- Python 3.6+
- Internet connection for API calls to the LiteLLM model.

### Libraries
The following Python libraries are required:
- `requests`: For making API calls to the LiteLLM server.
- `json`: For handling JSON data (part of the Python standard library).
- `random`: For random number generation (part of the Python standard library).

You can install `requests` using:
```bash
pip install requests
```

### LiteLLM, Ollama, and Autogen Setup
1. **LiteLLM**: This system uses the LiteLLM server to handle API requests for language model completions. Ensure the LiteLLM server is running locally and accessible through the `LITELLM_BASE_URL` in the code. 

2. **Ollama**: Ollama is a model package used in conjunction with LiteLLM. In this project, `"ollama/llama3.2:latest"` is used as the model configuration for the agents to interact with the organizer and contribute their responses.

3. **Autogen**: Autogen is an additional library that can provide automated processes and generation capabilities within the project. Initialize it as needed in your LiteLLM configuration for customized workflows.

The default LiteLLM configuration in the code is:
```python
LITELLM_BASE_URL = "http://0.0.0.0:4000"  # Modify if the server is hosted elsewhere
MODEL_NAME = "ollama/llama3.2:latest"
```

## Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/multi-agent-discussion-system.git
   cd multi-agent-discussion-system
   ```

2. **Run the Program**:
   Execute the main script:
   ```bash
   python main.py
   ```

3. **Interactive Mode**:
   - **Enter a Question**: The program prompts you to enter a question.
   - **Discussion Process**: The organizer initializes a multi-agent discussion, iteratively managing the discussion until it is satisfied with the response.
   - **Follow-Up Questions**: After each response, you’ll have the option to enter follow-up questions.

## System Workflow
1. **Organizer Receives Question**: Determines question difficulty and identifies relevant areas of study.
2. **Assigns Specialized Agents**: Selects and assigns agents based on the study areas identified.
3. **Iterative Discussion**: Agents provide responses, the organizer summarizes and checks satisfaction, and the process repeats if needed.
4. **Support for Follow-Up Questions**: Allows for continued discussions with follow-up questions until completion.

## Example Usage
Sample interaction:
```plaintext
Enter your question: Explain the relationship between gravity and time.
Organizer has determined the difficulty of the question is 'hard'.
Organizer has identified the following areas of study: Physicist, Mathematician
Organizer is not satisfied with the answer. Agents will discuss again.
...
Final Summary from Organizer: ...
Do you have a follow-up question? (Enter 'yes' to continue or 'no' to exit): yes
Enter your follow-up question: How does this relate to Einstein's theory?
...
```

## Future Enhancements
1. **Improved Question Analysis**: Integrate NLP for deeper understanding.
2. **Enhanced Satisfaction Checks**: Add a more robust satisfaction algorithm.
3. **Adaptive Agent Behavior**: Allow agents to refine answers based on organizer feedback.

## License
This project is licensed under the MIT License.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.
