import sqlite3
import bcrypt
import re
from fastapi import FastAPI, HTTPException, File, UploadFile, Depends
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from fastapi.security import HTTPBasicCredentials, HTTPBasic

app = FastAPI()

security = HTTPBasic()

# SQLite connection
def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

# Create users table
def create_users_table():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

create_users_table()

# Model for user registration
class User(BaseModel):
    username: str
    password: str

# Load the trained MobileNetV2 model
model = load_model('fashion_mnist_mobilenet.h5')

# Class labels for Fashion MNIST
class_labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Username and Password Validation Functions
def validate_username(username):
    """Validate username: Only lowercase alphabets, numbers, and underscores. Must start with a letter."""
    if re.match("^[a-z][a-z0-9_]*$", username):
        return True
    return False

def validate_password(password):
    """Validate password: At least 1 uppercase, 1 lowercase, 1 digit, 1 special character, and length >= 8."""
    if re.match(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$', password):
        return True
    return False

@app.post("/register")
def register(user: User):
    # Validate username and password
    if not validate_username(user.username):
        raise HTTPException(status_code=400, detail="Invalid username. Must start with a lowercase letter and can contain only lowercase letters, numbers, and underscores.")
    
    if not validate_password(user.password):
        raise HTTPException(status_code=400, detail="Invalid password. Must contain at least one uppercase letter, one lowercase letter, one digit, one special character, and be at least 8 characters long.")

    conn = get_db_connection()
    cursor = conn.execute("SELECT * FROM users WHERE username = ?", (user.username,))
    existing_user = cursor.fetchone()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists.")

    # Hash the password
    hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt())

    # Store the username and hashed password in SQLite
    conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (user.username, hashed_password.decode('utf-8')))
    conn.commit()
    conn.close()

    return {"message": "User registered successfully!"}

@app.post("/signin")
def signin(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    password = credentials.password

    conn = get_db_connection()
    cursor = conn.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    # Verify password
    stored_hashed_password = user['password'].encode('utf-8')
    if not bcrypt.checkpw(password.encode('utf-8'), stored_hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    return {"message": "User signed in successfully!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), credentials: HTTPBasicCredentials = Depends(security)):
    # Authenticate user
    username = credentials.username
    password = credentials.password

    conn = get_db_connection()
    cursor = conn.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    # Verify password
    stored_hashed_password = user['password'].encode('utf-8')
    if not bcrypt.checkpw(password.encode('utf-8'), stored_hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    # Open the uploaded image
    try:
        image = Image.open(file.file).convert("L")  # Convert to grayscale
        image = image.resize((32, 32))  # Resize to 32x32 for MobileNet
        image = np.array(image).reshape(1, 32, 32, 1).astype('float32') / 255.0
        image = np.repeat(image, 3, axis=-1)  # Duplicate grayscale to RGB (3 channels)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image format.")

    # Make prediction
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    return {"class": class_labels[predicted_class], "confidence": float(confidence)}

# Test endpoint
@app.get("/")
def root():
    return {
        "message": "Welcome to the FastAPI Fashion MNIST Model with SQLite Authentication!",
        "endpoints": {
            "/register": "POST - Register a new user with a valid username and password. Username must start with a lowercase letter and contain only lowercase letters, numbers, or underscores. Password must include at least one uppercase letter, one lowercase letter, one digit, one special character, and have a length of 8 or more.",
            "/signin": "POST - Sign in with your registered username and password.",
            "/predict": "POST - Authenticate with your registered username and password to upload an image and get the predicted class."
        }
    }
