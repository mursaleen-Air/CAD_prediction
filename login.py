# login.py
import streamlit as st

# Dummy credentials for demo purposes
USER_CREDENTIALS = {
    "user1": "1",
    "admin": "1",
}

def login():
    st.title("Login Page")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")

    if login_button:
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.success("Login successful!")
            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Invalid username or password")

def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()
    st.success("Logged out successfully.")

