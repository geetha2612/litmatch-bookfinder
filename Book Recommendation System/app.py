from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import google.generativeai as genai
from gtts import gTTS
import os
import uuid


# ========== GEMINI API SETUP ==========
api_key = "AIzaSyDLZNa5kRqIMEoQ5u8PFWJPaK1RBEZUVMA"  # Replace with your real API key
genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Function to get book overview from Gemini
def get_book_overview(title, author):
    try:
        prompt = f"Tell about the book '{title}' by {author} in three lines."
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error fetching overview for '{title}': {e}")
        return "Overview not available."

# Load Pickle Data
def load_pickle(file_path):
    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

pivot_table_data = pd.read_pickle('D:/My Document/New folder (3)/Book Recommendation System/data/pivot_table_data.pkl')
book_data = pd.read_pickle('D:/My Document/New folder (3)/Book Recommendation System/data/book_data.pkl')
popular_books = pd.read_pickle('D:/My Document/New folder (3)/Book Recommendation System/data/popular_books.pkl')
similarity_score = load_pickle('D:/My Document/New folder (3)/Book Recommendation System/data/similarity_score.pkl')

# Check Data Load
if popular_books is None: print("Failed to load popular_books.")
if pivot_table_data is None: print("Failed to load pivot_table_data.")
if book_data is None: print("Failed to load book_data.")
if similarity_score is None: print("Failed to load similarity_score.")

# Flask App
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html',
                           book_name=list(popular_books['Book-Title'].values),
                           author=list(popular_books['Book-Author'].values),
                           image_url=list(popular_books['Image-URL-M'].values),
                           votes=list(popular_books['count'].values),
                           avg_rating=list(popular_books['mean'].values))

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_books', methods=['POST'])
def recommend():
    user_input = request.form.get('user_input')

    if user_input not in pivot_table_data.index:
        return render_template('recommend.html', data=[], overviews=[], error="Book not found. Please try another title.")

    index = np.where(pivot_table_data.index == user_input)[0][0]
    similar_items = sorted(list(enumerate(similarity_score[index])), key=lambda x: x[1], reverse=True)[1:5]

    data = []
    overviews = []
    for i in similar_items:
        item = []
        book_title = pivot_table_data.index[i[0]]
        temp_df = book_data[book_data['Book-Title'] == book_title]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))

        try:
            overview_response = model.generate_content(f"Tell about the book titled '{book_title}' in three lines.")
            overview_text = overview_response.text.strip()

            # gTTS audio file generation
            tts = gTTS(overview_text)
            filename = f"static/audio/{uuid.uuid4().hex}.mp3"
            tts.save(filename)

        except Exception as e:
            overview_text = "Overview not available due to an error."
            filename = None

        overviews.append((book_title, overview_text, filename))
        data.append(item)

    return render_template('recommend.html', data=data, overviews=overviews)

if __name__ == '__main__':
    app.run(debug=True)