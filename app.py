import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def init():
    # Initialize memory and conversation chain
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(k=10)
    if "conversation" not in st.session_state:
        model = "mixtral-8x7b-32768"  # Fixed model
        groq_chat = ChatGroq(
            groq_api_key=GROQ_API_KEY,  # Replace with your actual API key
            model_name=model
        )
        st.session_state.conversation = ConversationChain(
            llm=groq_chat,
            memory=st.session_state.memory,
            output_key="response"  # Explicit output key
        )
    
    # Initialize messages if not already initialized
    if "messages" not in st.session_state:
        st.session_state.messages = []

# Main Streamlit app
def main():
    st.set_page_config(page_title="LangChain Chatbot", layout="wide")
    st.title("🤖 Ask Me Anything!")

    # Initialize memory and conversation
    init()

    # Chat Interface
    st.sidebar.header("Chatbot Settings")
    st.sidebar.write("Memory length: 10 (fixed)")

    # Display the chat history in bubbles
    st.markdown(
        """
        <style>
            .chat-container {
                display: flex;
                flex-direction: column;
                justify-content: flex-end;
                height: 70vh;
                overflow-y: auto;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 10px;
                # background-color: #f9f9f9;
            }
            .message {
                padding: 10px;
                border-radius: 10px;
                margin: 5px;
                max-width: 70%;
                display: flex;
                align-items: center;
            }
            .user {
                color:#00000,
                background-color: #DCF8C6;
                align-self: flex-start;
            }
            .bot {
                # color:#00000,
                background-color: #E5E5E5;
                align-self: flex-end;
            }
            .message img {
                border-radius: 50%;
                width: 30px;
                height: 30px;
                margin: 0 10px;
            }
            .input-box {
                position: fixed;
                bottom: 10px;
                left: 10%;
                width: 80%;
                display: flex;
                justify-content: center;
            }
        </style>
        """, unsafe_allow_html=True
    )

    # Chat container for messages
    chat_container = st.container()

    # Display chat history
    with chat_container:
        for sender, message in st.session_state.messages:
            if sender == "user":
                st.markdown(f'<div class="message user"><img src="https://img.icons8.com/ios/452/user-male-circle.png" /> <div>{message}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="message bot"><img src="https://cdn3.iconfinder.com/data/icons/chat-bot-emoji-blue-filled-color/300/14134081Untitled-3-4096.png" /> <div>{message}</div></div>', unsafe_allow_html=True)

    # User input field at the bottom like ChatGPT
    with st.container():
        user_input = st.text_input("You: ", placeholder="Type your message and press Enter", key="user_input", label_visibility="collapsed")

        if user_input:
            try:
                # Get response from the model
                response = st.session_state.conversation.run(user_input)
                # Append to chat history
                st.session_state.messages.append(("user", user_input))
                st.session_state.messages.append(("bot", response))
            except ValueError as e:
                st.error(f"Error: {e}")

    st.write("\n---")

if __name__ == "__main__":
    main()
