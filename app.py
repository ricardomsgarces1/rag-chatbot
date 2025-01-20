# Import necessary modules
import gradio as gr
import json
import time

# LangChain and related libraries for AI and tools
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import END

# Cisco Documentation sections available for loading and context. Additional sections could be added after running the scrapper script.
documentation_sections = [
    {
        "name": "Smart Bonding Partner API",
        "url": "https://developer.cisco.com/docs/smart-bonding-partner-api/",
        "file": "urls/smart-bonding-partner-api.json"
    },
    {
        "name": "Meraki Dashboard API",
        "url": "https://developer.cisco.com/meraki/api-v1/",
        "file": "urls/meraki-api.json"
    }
]

# Function to load URLs and contexts from JSON files
def load_urls_with_context(selected_section):
    """
    Load URLs and associated context based on the selected documentation section.

    Args:
        selected_section (str): Name of the documentation section.

    Returns:
        list: A list of dictionaries containing URL and context.
    """
    base_url, json_file_path = None, None

    # Match the selected section to the appropriate documentation entry
    for documentation in documentation_sections:
        if documentation['name'] == selected_section:
            base_url, json_file_path = documentation['url'], documentation['file']
            break

    if not base_url:
        return False

    # Load JSON data from the file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Construct URL-context pairs
    return [{"url": f"{base_url}{entry['url']}/", "context": entry['name']} for entry in data['urls']]

# Function to load documents from web sources
def load_documents(urls_with_context):
    """
    Load and parse documents from the provided URLs.

    Args:
        urls_with_context (list): List of URL-context dictionaries.

    Returns:
        list: A list of loaded documents with metadata.
    """
    documents = []
    for item in urls_with_context:
        try:
            loader = WebBaseLoader(web_paths=[item["url"]])
            docs = loader.load()
            for doc in docs:
                doc.metadata["context"] = item["context"]
                documents.append(doc)
        except Exception as e:
            print(f"Error loading {item['url']}: {e}")
    return documents

# Function to set up a FAISS vector store for document similarity
def setup_vectorstore(documents):
    """
    Process documents into embeddings and create a FAISS vector store.

    Args:
        documents (list): List of documents to process.

    Returns:
        FAISS: A FAISS vector store for similarity search.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    return FAISS.from_documents(splits, embedding=OpenAIEmbeddings())

# Tool definition for retrieval
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """
    Retrieve relevant documents based on a query.

    Args:
        query (str): Search query.

    Returns:
        tuple: Serialized response and retrieved documents.
    """
    retrieved_docs = vectorstore.similarity_search(query, k=2)
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}" for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# Function to handle user queries or generate responses
def query_or_respond(state: MessagesState):
    """
    Process user messages and determine the next action.

    Args:
        state (MessagesState): State of the conversation.

    Returns:
        dict: Updated conversation messages.
    """
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Function to generate a response using retrieved content
def generate(state: MessagesState):
    """
    Generate a response based on retrieved content and conversation history.

    Args:
        state (MessagesState): State of the conversation.

    Returns:
        dict: Generated messages.
    """
    recent_tool_messages = [
        message for message in reversed(state["messages"]) if message.type == "tool"
    ][::-1]

    docs_content = "\n\n".join(doc.content for doc in recent_tool_messages)
    system_message_content = (
        "You are an expert AI assistant with knowledge of Cisco's Smart Bonding Partner API and "
        "should only respond to questions related to it. Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know.\n\n"
        f"{docs_content}"
    )

    conversation_messages = [
        message for message in state["messages"]
        if message.type in ("human", "system") or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages
    response = llm.invoke(prompt)
    return {"messages": [response]}

# Function to initialize components and configure the graph
def initialize_components(selected_section):
    """
    Initialize components and graph nodes based on the selected documentation section.

    Args:
        selected_section (str): Name of the documentation section.
    """
    urls_with_context = load_urls_with_context(selected_section)
    documents = load_documents(urls_with_context)
    global vectorstore
    vectorstore = setup_vectorstore(documents)

    # Build the graph
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(ToolNode([retrieve]))
    graph_builder.add_node(generate)
    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges("query_or_respond", tools_condition, {END: END, "tools": "tools"})
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    memory = MemorySaver()
    global graph
    graph = graph_builder.compile(checkpointer=memory)

# Generate a unique session ID
def generate_session_id():
    """Generate a unique session ID based on the current time."""
    current_time = time.time()  # Get the current time in seconds since the epoch
    return str(int(current_time * 1000))  # Multiply by 1000 to get milliseconds and convert to string

# Example usage
generate_session_id()

# Initialize the LLM and components
llm = ChatOpenAI(model="gpt-4o-mini")
initialize_components(documentation_sections[0]['name'])

# Define the Gradio interface using a block structure
with gr.Blocks(theme=gr.themes.Base()) as demo:

    # Create a state to store the session ID
    session_id = gr.State()
    print(f"Initial session ID: {session_id.value}")  # Log the session ID for tracking

    # Add a header with information about the chatbot
    with gr.Row():
        gr.Markdown(
            """
            **Cisco API Chatbot PoC**  
            This chatbot is a **proof of concept (PoC)** to evaluate the feasibility of using retrieval-augmented generation (RAG) based on publicly available DevNet documentation for Cisco APIs.  
            Please note:  
            - The information provided by this chatbot may be **outdated** or **incorrect**.  
            - For accurate and up-to-date information, always refer to the official Cisco Developer documentation: [Cisco DevNet Docs](https://developer.cisco.com/docs/).  
            """
        )
        
    # Row for selecting the documentation section
    with gr.Row():
        section_selector = gr.Dropdown(
            label="Cisco Documentation Section",  # Dropdown label
            choices=[item["name"] for item in documentation_sections],  # List of section names as options
            value=documentation_sections[0]["name"]  # Default selected value
        )

    # Row to display a loading message, initially hidden
    with gr.Row(visible=False) as markdown_row:
        loading_message = gr.Markdown(
            value="Loading documentation, please wait... this might take a while!"
        )

    # Row for the chatbot interface, initially visible
    with gr.Row() as chatbot_row:
        chatbot = gr.Chatbot(type="messages")  # Chatbot widget to display conversations

    # Row for the input textbox where the user enters questions
    with gr.Row() as textbot_row:
        msg = gr.Textbox(label="Question")  # Textbox labeled "Question"

    # Row for the clear button to reset input and chat history
    with gr.Row() as button_row:
        clear = gr.ClearButton([msg, chatbot])  # Clear button resets both textbox and chatbot

    # Define the inference function to handle user input and generate responses
    def infere(user_input, session_id):
        """
        Perform inference using the conversation graph.

        Args:
            user_input (str): User's input question.

        Returns:
            str: Assistant's response.
        """
        result = None
        for step in graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},  # User input is sent as a message
            stream_mode="values",  # Enable streaming mode for responses
            config = {"configurable": {"thread_id": session_id.value}}  # Pass the session configuration
        ):
            result = step["messages"][-1].content  # Extract the last message content
            step["messages"][-1].pretty_print()  # Print the message for debugging/visualization
        return result

    # Function to handle user input and update chat history
    def respond(message, chat_history, session_id):
        """
        Process user input and generate a chatbot response.

        Args:
            message (str): User's input message.
            chat_history (list): Chat history containing past messages.
            session_id (str): Session ID.

        Returns:
            tuple: Cleared input box, updated chat history and Session ID.
        """

        # Check if the session ID is set
        if(session_id == None):
            session_id = gr.State(value=generate_session_id()) # Generate a session ID
        print(f"Current session ID: {session_id.value}")  # Log the session ID for tracking

        bot_response = infere(message, session_id)  # Get the assistant's response
        chat_history.append({"role": "user", "content": message})  # Add user message to history
        chat_history.append({"role": "assistant", "content": bot_response})  # Add assistant's response
        return "", chat_history, session_id  # Clear input box and update chat history

    # Function to reload components when the documentation section is changed
    def update_and_reload(selected_section):
        """
        Update the selected section and reload components.

        Args:
            selected_section (str): Newly selected documentation section.

        Returns:
            tuple: Updated visibility states for various UI rows.
        """
        initialize_components(selected_section)  # Reinitialize components with the new section
        return (
            gr.Row(visible=False),  # Hide loading message row
            gr.Row(visible=True),  # Show chatbot row
            gr.Row(visible=True),  # Show text input row
            gr.Row(visible=True)   # Show clear button row
        )

    # Function to show the loading message while updating components
    def show_loading_message():
        """
        Display the loading message while hiding other components.

        Returns:
            tuple: Visibility states for loading and chatbot-related rows.
        """
        return (
            gr.Row(height=150, visible=True),  # Show loading message
            gr.Row(visible=False),  # Hide chatbot row
            gr.Row(visible=False),  # Hide text input row
            gr.Row(visible=False)   # Hide clear button row
        )

    # Bind the dropdown change event to display the loading message and reload components
    section_selector.change(
        fn=show_loading_message,  # First, show the loading message
        inputs=[],  # No inputs required
        outputs=[markdown_row, chatbot_row, textbot_row, button_row]  # Update visibility of rows
    ).then(
        fn=update_and_reload,  # Then, reload components based on the new section
        inputs=[section_selector],  # Pass the selected section as input
        outputs=[markdown_row, chatbot_row, textbot_row, button_row]  # Update row visibility states
    )

    # Bind the textbox submission event to the respond function
    msg.submit(
        respond,  # Function to call on submission
        [msg, chatbot, session_id],  # Inputs: Textbox content and chat history
        [msg, chatbot, session_id]   # Outputs: Cleared textbox and updated chat history
    )

# Launch the Gradio interface
demo.launch()
