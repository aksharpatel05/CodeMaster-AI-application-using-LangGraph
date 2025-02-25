import os
import streamlit as st
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from openai import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# Define the state structure for the graph
class ReviewState(TypedDict):
    topic: str              # User's input topic (optional if code is pasted)
    language: str           # Programming language extracted or selected
    code: str               # Generated or user-pasted code
    test_cases: str         # Generated test cases
    validation_result: str  # Result of validation (pass/fail)
    peer_feedback: str      # Peer feedback (only for pasted code)
    documentation: str      # Final documentation (theory only)
    messages: Annotated[List[AIMessage], "List of messages for tracking"]
    model: str              # Selected model

# Prompt templates for each step
code_gen_prompt = ChatPromptTemplate.from_template(
    "Generate clean, error-free {language} code for the topic: {topic}. Include comments."
)
test_case_prompt = ChatPromptTemplate.from_template(
    "Generate test cases in {language} for the following code:\n{code}"
)
validation_prompt = ChatPromptTemplate.from_template(
    "Validate this {language} code:\n{code}\nusing these test cases:\n{test_cases}\nReturn 'pass' or 'fail' with explanation."
)
feedback_prompt = ChatPromptTemplate.from_template(
    "Review this {language} code:\n{code}\nand test results:\n{validation_result}\nProvide peer feedback as if from a colleague."
)
update_code_prompt = ChatPromptTemplate.from_template(
    "Based on this peer feedback:\n{peer_feedback}\nUpdate the following {language} code:\n{code}\nProvide the improved code with explanations."
)
doc_prompt = ChatPromptTemplate.from_template(
    "Given this {language} code:\n{code}\nWrite theoretical documentation for it. Include purpose, usage, performance(space and time complexity) and examples, but do not include the actual code in the output."
)

# Helper function to initialize LLM based on provider
def initialize_llm(provider: str, api_key: str, model: str):
    if provider == "Groq":
        return ChatGroq(model=model, api_key=api_key, temperature=0.7)
    elif provider == "OpenAI":
        return OpenAI(api_key=api_key)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

# Helper function to invoke LLM with a prompt and input
def invoke_llm(llm, provider: str, prompt, model: str, **kwargs):
    formatted_prompt = prompt.format(**kwargs)
    if provider == "Groq":
        return llm.invoke(formatted_prompt).content
    elif provider == "OpenAI":
        response = llm.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": formatted_prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    else:
        raise ValueError(f"Unsupported provider: {provider}")

# Step 1: Gather input (topic or pasted code)
def gather_input(state: ReviewState) -> ReviewState:
    state["topic"] = st.session_state.get("topic_input", "")
    state["code"] = st.session_state.get("code_input", "")
    state["language"] = st.session_state.get("selected_language", "Python")
    state["test_cases"] = ""
    state["validation_result"] = ""
    state["peer_feedback"] = ""
    state["documentation"] = ""

    if state["code"] and "selected_language" not in st.session_state:
        for lang in ["Java", "Python", "C++", "JavaScript"]:
            if lang.lower() in state["topic"].lower() or ("public class" in state["code"] and lang == "Java"):
                state["language"] = lang
                break
    elif state["topic"] and not state["code"]:
        for lang in ["Java", "Python", "C++", "JavaScript"]:
            if lang.lower() in state["topic"].lower():
                state["language"] = lang
                break

    state["messages"] = [HumanMessage(content=f"Input gathered: Topic={state['topic']}, Language={state['language']}")]
    return state

# Step 2: Generate code (for topic input)
def generate_code(state: ReviewState, llm, provider: str) -> ReviewState:
    if state["topic"] and not state["code"]:
        code = invoke_llm(llm, provider, code_gen_prompt, state["model"], language=state["language"], topic=state["topic"])
        state["code"] = code
        state["messages"].append(AIMessage(content=f"Generated {state['language']} code:\n{code}"))
    return state

# Step 3: Generate test cases
def generate_test_cases(state: ReviewState, llm, provider: str) -> ReviewState:
    if state["code"]:
        test_cases = invoke_llm(llm, provider, test_case_prompt, state["model"], language=state["language"], code=state["code"])
        state["test_cases"] = test_cases
        state["messages"].append(AIMessage(content=f"Generated {state['language']} test cases:\n{test_cases}"))
    return state

# Step 4: Validate code
def validate_code(state: ReviewState, llm, provider: str) -> ReviewState:
    if state["test_cases"]:
        result = invoke_llm(llm, provider, validation_prompt, state["model"], language=state["language"], code=state["code"], test_cases=state["test_cases"])
        state["validation_result"] = result
        state["messages"].append(AIMessage(content=f"Validation result:\n{result}"))
    return state

# Step 5: Peer feedback (for pasted code)
def peer_feedback(state: ReviewState, llm, provider: str) -> ReviewState:
    if not state["topic"] and state["code"]:
        feedback = invoke_llm(llm, provider, feedback_prompt, state["model"], language=state["language"], code=state["code"], validation_result=state["validation_result"] or "Not validated")
        state["peer_feedback"] = feedback
        state["messages"].append(AIMessage(content=f"Peer feedback:\n{feedback}"))
    return state

# Step 6: Update code based on feedback (for pasted code)
def update_code_based_on_feedback(state: ReviewState, llm, provider: str) -> ReviewState:
    if state["peer_feedback"] and not state["topic"]:
        updated_code = invoke_llm(llm, provider, update_code_prompt, state["model"], language=state["language"], code=state["code"], peer_feedback=state["peer_feedback"])
        state["code"] = updated_code
        state["messages"].append(AIMessage(content=f"Updated {state['language']} code based on feedback:\n{updated_code}"))
    return state

# Step 7: Generate documentation
def generate_documentation(state: ReviewState, llm, provider: str) -> ReviewState:
    if state["code"]:
        docs = invoke_llm(llm, provider, doc_prompt, state["model"], language=state["language"], code=state["code"])
        state["documentation"] = docs
        state["messages"].append(AIMessage(content=f"Documentation:\n{docs}"))
    return state

# Build the LangGraph workflow based on input type
def build_workflow(topic_mode: bool, provider: str):
    workflow = StateGraph(ReviewState)
    workflow.add_node("gather_input", gather_input)
    workflow.add_edge(START, "gather_input")

    last_node = "gather_input"

    if topic_mode:
        # Topic mode: Code -> Test Cases -> Validation -> Documentation
        workflow.add_node("generate_code", lambda state: generate_code(state, st.session_state.llm, provider))
        workflow.add_edge(last_node, "generate_code")
        last_node = "generate_code"
    else:
        # Paste mode: Test Cases -> Validation -> Feedback -> Update Code -> Documentation
        last_node = "gather_input"

    workflow.add_node("generate_test_cases", lambda state: generate_test_cases(state, st.session_state.llm, provider))
    workflow.add_edge(last_node, "generate_test_cases")
    last_node = "generate_test_cases"

    workflow.add_node("validate_code", lambda state: validate_code(state, st.session_state.llm, provider))
    workflow.add_edge(last_node, "validate_code")
    last_node = "validate_code"

    if not topic_mode:
        workflow.add_node("generate_peer_feedback", lambda state: peer_feedback(state, st.session_state.llm, provider))
        workflow.add_edge(last_node, "generate_peer_feedback")
        last_node = "generate_peer_feedback"

        workflow.add_node("update_code_based_on_feedback", lambda state: update_code_based_on_feedback(state, st.session_state.llm, provider))
        workflow.add_edge(last_node, "update_code_based_on_feedback")
        last_node = "update_code_based_on_feedback"

    workflow.add_node("generate_documentation", lambda state: generate_documentation(state, st.session_state.llm, provider))
    workflow.add_edge(last_node, "generate_documentation")
    last_node = "generate_documentation"

    workflow.add_edge(last_node, END)

    return workflow.compile()

# Model options and mapping
MODEL_OPTIONS = [
    "llama3-70b-8192 (Open Source) - Groq",
    "llama-3.3-70b-versatile (Open Source) - Groq",
    "gpt-4 (Paid) - OpenAI",
    "gpt-2 (Open Source) - OpenAI"
]
MODEL_MAP = {
    "llama3-70b-8192 (Open Source) - Groq": ("llama3-70b-8192", "Groq"),
    "llama-3.3-70b-versatile (Open Source) - Groq": ("llama-3.3-70b-versatile", "Groq"),
    "gpt-4 (Paid) - OpenAI": ("gpt-4", "OpenAI"),
    "gpt-2 (Open Source) - OpenAI": ("gpt-2", "OpenAI")
}

# Streamlit UI
def main():
    st.set_page_config(page_title="CodeMaster AI", layout="wide")
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>CodeMaster AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Your intelligent coding assistant for generating, testing, and improving code.</p>", unsafe_allow_html=True)

    # Model selection
    with st.container():
        st.markdown("### 🚀 Step 1: Select Your Model")
        selected_model_display = st.selectbox("Choose a model", MODEL_OPTIONS, key="selected_model")
        selected_model, provider = MODEL_MAP[selected_model_display]

    # API key input
    with st.container():
        st.markdown(f"### 🔑 Step 2: Enter {provider} API Key")
        api_key = st.text_input(f"Enter your {provider} API key", type="password", key="api_key")
        if not api_key:
            st.error(f"Please enter a valid {provider} API key.")
            return

    # Initialize LLM
    if "llm" not in st.session_state or st.session_state.get("prev_api_key") != api_key or st.session_state.get("prev_model") != selected_model:
        try:
            st.session_state.llm = initialize_llm(provider, api_key, selected_model)
            st.session_state.prev_api_key = api_key
            st.session_state.prev_model = selected_model
            st.session_state.provider = provider
        except Exception as e:
            st.error(f"Failed to initialize {provider} client: {str(e)}. Check your API key.")
            return

    # Input options
    st.markdown("### ✍️ Step 3: Provide Input")
    input_option = st.radio("Choose how to start:", ["Enter a Problem Topic", "Paste Existing Code"], key="input_option")

    if input_option == "Enter a Problem Topic":
        st.text_input("Enter a problem topic (e.g., 'write a Python code for merge sort')", key="topic_input")
        st.session_state.code_input = ""
    else:
        st.text_area("Paste your code here", key="code_input", height=150)
        st.selectbox("Select Language", ["Python", "Java", "C++", "JavaScript"], key="selected_language")
        st.session_state.topic_input = ""

    # Run button
    if st.button("Generate Results", key="run_button", help="Click to process your input"):
        if (input_option == "Enter a Problem Topic" and not st.session_state.get("topic_input")) or \
           (input_option == "Paste Existing Code" and not st.session_state.get("code_input")):
            st.error("Please provide a topic or paste code.")
            return

        with st.spinner("Processing your request..."):
            app = build_workflow(input_option == "Enter a Problem Topic", st.session_state.provider)
            initial_state = {"messages": [], "model": selected_model}
            try:
                result = app.invoke(initial_state)
                
                # Display results in an attractive UI
                st.markdown("### ✨ Results")
                
                if input_option == "Enter a Problem Topic" and result.get("topic"):
                    st.markdown(f"**Topic:** *{result['topic']}*")
                
                if result.get("code"):
                    expander_title = f"{result['language']} Code"
                    if input_option == "Paste Existing Code" and result.get("peer_feedback"):
                        expander_title += " (Updated Based on Feedback)"
                    with st.expander(expander_title, expanded=True):
                        st.code(result["code"], language=result["language"].lower())
                
                if result.get("test_cases"):
                    with st.expander("Test Cases", expanded=True):
                        st.code(result["test_cases"], language=result["language"].lower())
                
                if result.get("validation_result"):
                    with st.expander("Validation Report", expanded=True):
                        validation_result = result["validation_result"].lower()
                        st.markdown(f"**Result:** {'✅ Pass' if 'pass' in validation_result else '❌ Fail'}")
                        st.text(result["validation_result"])
                
                if input_option == "Paste Existing Code" and result.get("peer_feedback"):
                    with st.expander("Peer Feedback", expanded=True):
                        st.markdown(result["peer_feedback"])
                
                if result.get("documentation"):
                    with st.expander("Documentation", expanded=True):
                        st.markdown(result["documentation"])
            
            except Exception as e:
                st.error(f"Error processing request: {str(e)}. Check your {provider} API key and input.")

if __name__ == "__main__":
    main()