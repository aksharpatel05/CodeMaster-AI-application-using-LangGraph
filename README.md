# CodeMaster-AI-application-using-LangGraph


![CodeMaster AI Demo](demo.gif) <!-- Replace with an actual demo GIF or screenshot if you have one -->

**CodeMaster AI** is an intelligent coding assistant designed to streamline the process of generating, testing, and improving code. Whether you're starting with a problem idea or refining existing code, this tool automates the workflow—from code generation to documentation—using cutting-edge AI technologies.

---

## Features

### For Problem Topics
- Input a coding problem (e.g., "write a Python code for merge sort").
- Automatically generates:
  - Clean, commented code.
  - Test cases.
  - Validation report.
  - Theoretical documentation.

### For Paste/write Code
- Paste/write your existing code and select a language.
- Automatically generates:
  - Test cases.
  - Validation report.
  - Peer feedback (as if from a colleague).
  - Updated code based on feedback.
  - Theoretical documentation.

---

---

## Technologies Used
- **LangChain**: Drives intelligent prompt design and LLM interactions for code generation, testing, feedback, and documentation.
- **LangGraph**: The backbone of the workflow! LangGraph structures the pipeline as a directed graph, dynamically orchestrating tasks like validation, feedback, and code updates. Its state management makes complex workflows modular and scalable.
- **Streamlit**: Provides a sleek, interactive web UI.
- **Groq/OpenAI**: Powers the language models (e.g., LLaMA, GPT) behind the scenes.
- **Python**: Core language for implementation.

### Focus on LangGraph
LangGraph is key to this project’s flexibility. For example, the "Paste/write Code" flow—Test Cases → Validation → Feedback → Updated Code → Documentation—is a graph where each node (task) feeds into the next. This approach ensures seamless task chaining and easy extensibility.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/aksharpatel05/CodeMaster-AI-application-using-LangGraph.git
   cd CodeMaster-AI-application-using-LangGraph

2. **Install Dependencies**:
- Required packages: streamlit, langchain, langgraph, langchain-groq, openai.

3. **Set Up API Keys**:
- Obtain a Groq or OpenAI API key.
- Enter it in the app when prompted.

4. **Run the App**:
```bash
streamlit run app.py
