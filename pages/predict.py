import streamlit as st
import requests
import base64
import io
from PIL import Image
from streamlit_option_menu import option_menu

# FastAPI backend URL
backend_url = "http://localhost:8001"

def main():
    st.title("Predict")

    if "username" in st.session_state and "password" in st.session_state:
        username = st.session_state["username"]
        password = st.session_state["password"]

        # Add user icon with sign-out option
        user_menu = option_menu(
            None, [f"👤 {username}", "Sign Out"],
            icons=['person', 'box-arrow-right'],
            menu_icon="cast", default_index=0, orientation="horizontal"
        )

        if user_menu == "Sign Out":
            st.session_state.pop("username", None)
            st.session_state.pop("password", None)
            st.switch_page("frontend.py")

        uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_container_width=True)

            if st.button("Predict"):
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                image_bytes = buffer.getvalue()
                encoded_image = base64.b64encode(image_bytes).decode()

                headers = {"Authorization": f"Basic {base64.b64encode(f'{username}:{password}'.encode()).decode()}"}
                files = {"file": (uploaded_file.name, io.BytesIO(base64.b64decode(encoded_image)), "image/png")}
                response = requests.post(f"{backend_url}/predict", files=files, headers=headers)

                if response.status_code == 200:
                    prediction = response.json()
                    st.success(f"Predicted Class: {prediction['class']}")
                    st.info(f"Confidence: {prediction['confidence']:.2f}")
                else:
                    st.error(response.json()["detail"])
    else:
        st.markdown(
            '<p style="color:red;">Please <a href="/" target="_self">sign in</a> to use the prediction feature.</p>',
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
