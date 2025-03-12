import streamlit as st
import requests
import time

# FastAPI backend URL
backend_url = "http://localhost:8001"

def main():
    st.title("Register")

    username = st.text_input("Username", help="Username must start with a lowercase letter and can contain only lowercase letters, numbers, and underscores.")
    password = st.text_input("Password", type="password", help="Password must include at least one uppercase letter, one lowercase letter, one digit, one special character, and have a length of 8 or more.")

    if st.button("Register"):
        response = requests.post(f"{backend_url}/register", json={"username": username, "password": password})
        
        if response.status_code == 200:
            st.success("User registered successfully!")
            time.sleep(3)
            st.switch_page("frontend.py")
        else:
            st.error(response.json()["detail"])

if __name__ == "__main__":
    main()
