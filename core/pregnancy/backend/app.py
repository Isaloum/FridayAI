import os
import sys
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from FridayAI_SuperUltraBrilliant import FridayAI
    logger.info("✅ Successfully imported FridayAI core")
except ImportError as e:
    logger.error(f"❌ Failed to import FridayAI core: {e}")
    # Create dummy class for fallback
    class FridayAI:
        def handle_api_request(self, data):
            return {"content": "System error: Core module missing"}

app = Flask(__name__)
CORS(app)
friday_ai = FridayAI()

# Try to import agents
agents_loaded = False
try:
    from agents import TestAgent  # Adjust based on your agent names
    friday_ai.add_agent(TestAgent())
    agents_loaded = True
    logger.info("✅ Agents loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Could not load agents: {e}")

@app.route('/api/friday', methods=['POST'])
def chat_endpoint():
    try:
        data = request.json
        logger.debug(f"Received request: {data}")
        response = friday_ai.handle_api_request(data)
        return jsonify(response)
    except Exception as e:
        logger.exception("Server error")
        return jsonify({
            "content": "⚠️ System error occurred. Please try again later.",
            "error": str(e)
        }), 500

@app.route('/health')
def health_check():
    return jsonify({
        "status": "ok",
        "agents_loaded": agents_loaded,
        "core_loaded": "FridayAI" in globals()
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)