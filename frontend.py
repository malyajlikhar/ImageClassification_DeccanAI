import streamlit as st
import requests
import base64

# FastAPI backend URL
backend_url = "http://localhost:8000"

def main():
    st.title("Sign In")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Sign In"):
        headers = {"Authorization": f"Basic {base64.b64encode(f'{username}:{password}'.encode()).decode()}"}
        response = requests.post(f"{backend_url}/signin", headers=headers)
        
        if response.status_code == 200:
            st.success("User signed in successfully!")
            st.session_state["username"] = username
            st.session_state["password"] = password
            st.switch_page("pages/predict.py")
        else:
            st.error(response.json()["detail"])

    if st.button("Not registered? Sign Up"):
        st.switch_page("pages/register.py")

if __name__ == "__main__":
    main()
