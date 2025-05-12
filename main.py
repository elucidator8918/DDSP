"""
AInstrument Main FastAPI Application

This file contains the FastAPI application setup and route definitions.
"""
import os
import time
from typing import Dict, Optional
import json
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
import pyrebase
import requests
from pyngrok import ngrok

from music_conversion import MusicConversion
from vocal_extraction import VocalExtraction

# Initialize FastAPI app
app = FastAPI(title="AInstrument API", 
              description="AI-powered musical instrument conversion API",
              version="1.0.0")

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Firebase configuration
firebase_config = {
    "apiKey": "AIzaSyAcbBkzH2YnTPVyDhKGjeA7EFAQf3wNTeE",
    "authDomain": "ainstrument-a03f0.firebaseapp.com",
    "databaseURL": "https://ainstrument-a03f0-default-rtdb.firebaseio.com",
    "storageBucket": "ainstrument-a03f0.appspot.com",
}

# Initialize Firebase
firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()
db = firebase.database()
storage = firebase.storage()

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("extracted_vocals", exist_ok=True)
os.makedirs("output_songs", exist_ok=True)

@app.get("/")
async def root():
    """API health check endpoint"""
    return {"status": "ok", "message": "AInstrument API is running"}

@app.post("/api/v1/extract-vocal")
async def extract_vocal(
    songName: str = Form(...),
    instrument: str = Form(...),
    audio: UploadFile = File(...),
    auth: Optional[str] = Header(None)
):
    """Extract vocals from an audio file and convert to instrument"""
    if not auth:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        # Verify authentication token
        user_info = verify_auth_token(auth)
        user_uid = user_info["localId"]
        
        # Create timestamped directory
        upload_time = str(time.time())
        upload_dir = os.path.join("uploads", upload_time)
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save uploaded file
        song_path = os.path.join(upload_dir, audio.filename)
        with open(song_path, "wb") as f:
            f.write(await audio.read())
        
        # Extract vocals
        vocal_extractor = VocalExtraction(song_path, songName)
        vocal_extractor.extract_vocal()
        extracted_path = vocal_extractor.destination_path
        
        # Upload original song to Firebase
        upload_url = upload_to_firebase(
            song_path, 
            f"uploadedSongs/{upload_time}/{audio.filename}", 
            auth
        )
        
        # Convert vocals to instrument
        music_converter = MusicConversion(songName, instrument, extracted_path)
        converted_path = music_converter.load_song_and_extract_features()
        
        # Upload converted song to Firebase
        extracted_url = upload_to_firebase(
            converted_path, 
            f"extractedSongs/{upload_time}/{audio.filename}", 
            auth
        )
        
        # Save to user's profile in database
        save_to_user_profile(user_uid, audio.filename, extracted_url, upload_url, instrument, auth)
        
        return {
            "success": 1,
            "message": "Extraction and conversion completed successfully",
            "data": {
                "extractedSongUrl": extracted_url,
                "uploadedSongUrl": upload_url
            }
        }
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "success": 0,
            "message": f"Error processing request: {str(e)}"
        }

@app.post("/api/v1/signin-user")
async def signin_user(data: Dict):
    """Sign in a user with email and password"""
    try:
        email = data.get("email")
        password = data.get("password")
        
        if not email or not password:
            return {"success": 0, "message": "Email and password are required"}
        
        # Sign in user
        user = auth.sign_in_with_email_and_password(email, password)
        
        # Get account info
        account_info = auth.get_account_info(user.get("idToken"))
        
        # Check if email is verified
        if account_info.get("users")[0].get("emailVerified"):
            return {
                "success": 1,
                "message": "Logged in successfully",
                "data": {
                    "userUID": user.get("localId"),
                    "token": user.get("idToken"),
                    "email": user.get("email")
                }
            }
        else:
            return {
                "success": 0,
                "message": "Please verify email address to login"
            }
    
    except requests.exceptions.HTTPError as error:
        error_data = json.loads(error.args[1])
        error_message = error_data.get("error", {}).get("message", "Authentication failed")
        return {"success": 0, "message": error_message}

@app.post("/api/v1/create-user")
async def create_user(data: Dict):
    """Create a new user with email and password"""
    try:
        name = data.get("name")
        email = data.get("email")
        password = data.get("password")
        confirm_pass = data.get("confirmPass")
        
        if not name or not email or not password:
            return {"success": 0, "message": "Name, email, and password are required"}
            
        if password != confirm_pass:
            return {"success": 0, "message": "Passwords do not match"}
        
        # Create user
        user = auth.create_user_with_email_and_password(email, password)
        
        # Send verification email
        auth.send_email_verification(user["idToken"])
        
        # Save user data to database
        user_data = {"name": name, "email": email}
        db.child("users").child(user["localId"]).set(user_data, user["idToken"])
        
        return {
            "success": 1,
            "message": "Verification link has been sent to the email"
        }
    
    except requests.exceptions.HTTPError as error:
        error_data = json.loads(error.args[1])
        error_message = error_data.get("error", {}).get("message", "User creation failed")
        return {"success": 0, "message": error_message}

@app.get("/api/v1/get-profile-details")
async def get_profile_details(auth: Optional[str] = Header(None)):
    """Get user profile details"""
    if not auth:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        # Verify authentication token
        user_info = verify_auth_token(auth)
        user_uid = user_info["localId"]
        
        # Get user data
        user_data = db.child("users").child(user_uid).get(auth)
        
        return {
            "success": 1,
            "message": "Data fetched successfully",
            "data": user_data.val()
        }
    
    except requests.exceptions.HTTPError as error:
        error_data = json.loads(error.args[1])
        error_message = error_data.get("error", {}).get("message", "Failed to get profile")
        return {"success": 0, "message": error_message}

def verify_auth_token(token):
    """Verify Firebase authentication token"""
    try:
        user_info = auth.get_account_info(token)
        return user_info.get("users")[0]
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid authentication token: {str(e)}")

def upload_to_firebase(file_path, storage_path, token):
    """Upload a file to Firebase Storage"""
    try:
        storage.child(storage_path).put(file_path, token)
        return storage.child(storage_path).get_url(token)
    except Exception as e:
        raise Exception(f"Failed to upload to Firebase: {str(e)}")

def save_to_user_profile(user_uid, song_name, extracted_url, uploaded_url, instrument, token):
    """Save song details to user's profile"""
    try:
        count = db.child("users").child(user_uid).child("songs").shallow().get(token)
        song_data = {
            "songName": song_name,
            "extractedSongUrl": extracted_url,
            "uploadedSongUrl": uploaded_url,
            "thumbnailUrl": "",
            "instrument": instrument
        }
        
        if count.val():
            db.child("users").child(user_uid).child("songs").child(str(len(count.val()))).update(song_data, token)
        else:
            db.child("users").child(user_uid).child("songs").child("0").update(song_data, token)
    except Exception as e:
        raise Exception(f"Failed to save to user profile: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Open a HTTP tunnel on port 8000 (default for FastAPI)
    public_url = ngrok.connect(8000)
    print(f"FastAPI is accessible publicly at {public_url}")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)