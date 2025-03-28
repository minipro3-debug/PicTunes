from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np
import requests
import os
import json  # Add this import

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
db = SQLAlchemy(app)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Keep existing User model and API configurations
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

# Check if user is logged in
def is_logged_in():
    return session.get('username') is not None

# Image Captioning Setup
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base",use_fast=True)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

SPOTIFY_CLIENT_ID = "8e267334234f406f9a73e30d976398a3"
SPOTIFY_CLIENT_SECRET = "4da6203141c74520a170702d7cd6a2ad"
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET
))

MISTRAL_API_KEY = "r9DzjYYPDjBVsIZvD5dWBUAAghEVjLmq"

# Add these imports at the top
import logging
from datetime import datetime

# Add logging configuration after Flask initialization
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Update the generate_caption function with error handling
def generate_caption(img):
    try:
        img_input = Image.fromarray(img).convert("RGB")
        inputs = processor(img_input, return_tensors="pt")
        out = model.generate(**inputs, max_length=50)
        return processor.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Caption generation error: {str(e)}")
        return "Error generating caption"

# Update the generate_instagram_caption function
def generate_instagram_caption(base_caption, style, language):
    try:
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
        data = {
            "model": "mistral-small",
            "messages": [
                {"role": "system", "content": "You are a gen-z creative assistant that generates Instagram-worthy captions. Generate unique and good captions every time. Generate only one complete caption along with suitable emojis."},
                {"role": "user", "content": f"Generate a {style.lower()} caption in {language} for: {base_caption}"}
            ],
            "temperature": 0.8,
            "max_tokens": 1000  # Increased token limit
        }
        response = requests.post(url, json=data, headers=headers, timeout=10)
        response.raise_for_status()
        caption = response.json()["choices"][0]["message"]["content"].strip()
        return caption
    except Exception as e:
        logger.error(f"Instagram caption generation error: {str(e)}")
        return "Error generating creative caption"

# Update the get_spotify_recommendations function
def get_keywords(blip_caption):
    try:
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "mistral-small",
            "messages": [
                {"role": "system", "content": "You are an amazing AI assistant that extracts important keywords for song searches based on a given caption. Provide only a short list of keywords."},
                {"role": "user", "content": f"Extract and give only the top keywords that describe mood, theme, or emotions from this caption: {blip_caption}. Keep it concise."}
            ],
            "temperature": 0.7,
            "max_tokens": 50,
            "top_p": 0.9
        }
        response = requests.post(url, json=data, headers=headers, timeout=10)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Keyword extraction error: {str(e)}")
        return f"Error extracting keywords: {str(e)}"

def get_spotify_recommendations(blip_caption, cap_lang):
    try:
        search_caption = get_keywords(blip_caption)
        if "Error" in search_caption:
            logger.error(f"Keyword extraction failed: {search_caption}")
            return "<p>Could not generate song recommendations.</p>"
            
        search_query = f"{search_caption} {cap_lang}"
        results = sp.search(q=search_query, type="track", limit=20)
        
        if not results or "tracks" not in results:
            return "<p>No songs found.</p>"
            
        seen_tracks = set()
        filtered_songs = []
        
        for track in results["tracks"]["items"]:
            song_name = track["name"]
            artist_name = track["artists"][0]["name"]
            
            if (song_name, artist_name) not in seen_tracks:
                seen_tracks.add((song_name, artist_name))
                filtered_songs.append(track)
            
            if len(filtered_songs) == 10:
                break

        if not filtered_songs:
            return "<p>No songs found.</p>"

        formatted_songs = ""
        for track in filtered_songs:
            song_name = track["name"]
            artist_name = track["artists"][0]["name"]
            song_url = track["external_urls"]["spotify"]
            cover_url = track["album"]["images"][0]["url"] if track["album"]["images"] else ""
            
            formatted_songs += f"""
            <div style='display: flex; align-items: center; margin-bottom: 10px; border-bottom: 1px solid #ddd; padding-bottom: 10px;'>
                <img src='{cover_url}' alt='Cover' style='width: 60px; height: 60px; border-radius: 5px; margin-right: 10px;'>
                <div style='flex-grow: 1;'>
                    <p style='margin: 0;'><strong>{song_name}</strong> by {artist_name}</p>
                    <div style='display: flex; gap: 10px; margin-top: 5px;'>
                        <a href='{song_url}' target='_blank' class='spotify-btn'>▶ Listen</a>
                        <button onclick='saveSong({json.dumps({
                            "song_name": song_name,
                            "artist_name": artist_name,
                            "spotify_url": song_url,
                            "cover_url": cover_url
                        })})' class='save-song-btn'>
                            <i class='fas fa-heart'></i> Save
                        </button>
                    </div>
                </div>
            </div>
            """
        return formatted_songs
    except Exception as e:
        logger.error(f"Spotify recommendations error: {str(e)}")
        return "<p>Error getting song recommendations</p>"

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/dashboard')
def dashboard():
    if not session.get('username'):
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    if not session.get('username'):
        return jsonify({'error': 'Not authenticated'}), 401
        
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if not file.content_type.startswith('image/'):
        return jsonify({'error': 'File must be an image'}), 400
    
    try:
        image = Image.open(file.stream)
        image_array = np.array(image)
        
        style = request.form.get('style', 'Normal')
        language = request.form.get('language', 'English')
        
        base_caption = generate_caption(image_array)
        if "Error" in base_caption:
            return jsonify({'error': base_caption}), 500
            
        insta_caption = generate_instagram_caption(base_caption, style, language)
        recommendations = get_spotify_recommendations(base_caption,language)
        
        return jsonify({
            'caption': insta_caption,
            'base_caption': base_caption,  # Added base caption to response
            'recommendations': recommendations,
            'status': 'success'
        })
    except Exception as e:
        logger.error(f"Process image error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and check_password_hash(user.password, request.form['password']):
            session['username'] = user.username
            flash('Login successful', 'success')
            return redirect(url_for('dashboard'))  # Changed to redirect to dashboard
        flash('Invalid credentials', 'error')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username, email, password = request.form['username'], request.form['email'], request.form['password']
        if User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first():
            flash('User already exists', 'error')
        else:
            db.session.add(User(username=username, email=email, password=generate_password_hash(password)))
            db.session.commit()
            flash('Registration successful!', 'success')
            return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('Logged out', 'success')
    return redirect(url_for('welcome'))

# Remove the duplicate login route and gradio-app route since we're not using Gradio anymore

# Add after Flask initialization
app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # 30 minutes
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True

@app.route('/save_song', methods=['POST'])
def save_song():
    if not session.get('username'):
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.json
    user = User.query.filter_by(username=session['username']).first()
    
    existing_song = FavoriteSong.query.filter_by(
        user_id=user.id,
        song_name=data['song_name'],
        artist_name=data['artist_name']
    ).first()
    
    if existing_song:
        return jsonify({'message': 'Song already in favorites'}), 200
        
    new_favorite = FavoriteSong(
        user_id=user.id,
        song_name=data['song_name'],
        artist_name=data['artist_name'],
        spotify_url=data['spotify_url'],
        cover_url=data['cover_url']
    )
    
    db.session.add(new_favorite)
    db.session.commit()
    return jsonify({'message': 'Song saved to favorites'}), 200

@app.route('/favorites')
def favorites():
    if not session.get('username'):
        flash('Please login first', 'error')
        return redirect(url_for('login'))
        
    user = User.query.filter_by(username=session['username']).first()
    favorite_songs = FavoriteSong.query.filter_by(user_id=user.id).all()
    return render_template('favorites.html', songs=favorite_songs)

class FavoriteSong(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    song_name = db.Column(db.String(200), nullable=False)
    artist_name = db.Column(db.String(200), nullable=False)
    spotify_url = db.Column(db.String(300), nullable=False)
    cover_url = db.Column(db.String(300), nullable=False)

@app.route('/remove_favorite/<int:song_id>', methods=['POST'])
def remove_favorite(song_id):
    if not session.get('username'):
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        user = User.query.filter_by(username=session['username']).first()
        song = FavoriteSong.query.filter_by(id=song_id, user_id=user.id).first()
        
        if song:
            db.session.delete(song)
            db.session.commit()
            return jsonify({'message': 'Song removed from favorites'}), 200
        else:
            return jsonify({'error': 'Song not found'}), 404
            
    except Exception as e:
        logger.error(f"Error removing favorite: {str(e)}")
        return jsonify({'error': 'Failed to remove song'}), 500

# Add after Flask initialization
app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # 30 minutes
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000, debug=False)
