"""
====================================================================
🧠 FridayAI SuperUltraBrilliant™ Backend API
📁 File: friday_ai_flask_api.py
🔧 Description: Flask-based API interface for the FridayAI logic engine
📌 Purpose: Connects frontend to the core AI logic (FridayAI class)
🧪 Status: Development Ready
👨‍💻 Author: [Your Name or Team Name]
====================================================================
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from fridayai import FridayAI  # This should point to your SuperUltraBrilliant logic file

app = Flask(__name__)
CORS(app)

# Initialize your AI system
from core.EmotionClassifier import EmotionClassifier
from core.MemoryCore import MemoryCore

memory = MemoryCore()
emotion = EmotionClassifier()
friday = FridayAI(memory, emotion)


# Log branding on startup
print("🔥 FridayAI SuperUltraBrilliant™ Backend Initialized!")

@app.route('/api/message', methods=['POST'])
def handle_message():
    data = request.json
    user_input = data.get('message', '')
    user_name = data.get('userName', '')
    pregnancy_week = data.get('pregnancyWeek', 0)
    tone = data.get('tone', 'supportive')

    if not user_input:
        return jsonify({'error': 'No message provided'}), 400

    # Process the message
    response = friday.process_message(user_input, user_name, pregnancy_week, tone)

    return jsonify({
        'title': 'FridayAI SuperUltraBrilliant™',
        'response': response
    })

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        'service': 'FridayAI SuperUltraBrilliant™',
        'version': '1.0.0',
        'status': 'ready',
        'author': 'Your Name or Team'
    })

if __name__ == '__main__':
    app.run(debug=True)
