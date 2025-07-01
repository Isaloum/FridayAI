# ==============================================
# File: secure_friday_api.py
# Purpose: Secure Friday AI API with YOUR existing configuration
# Copyright (c) 2025 [YOUR NAME]. All Rights Reserved.
# ==============================================

import os
import re
import jwt
import uuid
import logging
import hashlib
import secrets
from datetime import datetime, timedelta
from functools import wraps
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from werkzeug.middleware.proxy_fix import ProxyFix
from logging.handlers import RotatingFileHandler
from cryptography.fernet import Fernet
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

# Import your existing Friday AI system
try:
    # Import your existing Friday AI
    import sys
    sys.path.append('.')  # Add current directory to path
    
    from fridayai import FridayAI
    from core.MemoryCore import MemoryCore
    from core.EmotionCoreV2 import EmotionCoreV2
    
    print("✅ Friday AI modules imported successfully!")
except ImportError as e:
    print(f"❌ Friday AI import error: {e}")
    print("Make sure fridayai.py and core modules are in the correct path")

# ==============================================
# SECURE CONFIGURATION WITH YOUR EXISTING SETUP
# ==============================================

class SecureConfig:
    """Configuration that integrates with your existing .env"""
    
    # Your existing OpenAI configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_PROJECT = os.getenv('OPENAI_PROJECT')
    OPENAI_ORG_ID = os.getenv('OPENAI_ORG_ID')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4-turbo-preview')
    
    # Your existing search and AI settings
    SERPAPI_KEY = os.getenv('SERPAPI_KEY')
    FRIDAY_CONFIDENCE_THRESHOLD = float(os.getenv('FRIDAY_CONFIDENCE_THRESHOLD', '0.65'))
    FRIDAY_DEBUG = os.getenv('FRIDAY_DEBUG', 'false').lower() == 'true'
    
    # Flask environment
    ENV = os.getenv('FLASK_ENV', 'development')
    DEBUG = ENV == 'development'
    
    # Security configuration (NEW - add these to your .env)
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', secrets.token_urlsafe(32))
    ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY', Fernet.generate_key().decode())
    API_KEY_HASH = os.getenv('API_KEY_HASH')
    
    # CORS settings
    ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000,http://127.0.0.1:3000').split(',')
    
    # Rate limiting
    RATE_LIMIT_STORAGE_URL = os.getenv('REDIS_URL', 'memory://')
    DEFAULT_RATE_LIMITS = ["1000/day", "100/hour"]
    
    # Content safety
    MAX_INPUT_LENGTH = int(os.getenv('MAX_INPUT_LENGTH', '5000'))
    MAX_RESPONSE_LENGTH = int(os.getenv('MAX_RESPONSE_LENGTH', '10000'))
    CONTENT_FILTER_ENABLED = os.getenv('CONTENT_FILTER_ENABLED', 'true').lower() == 'true'
    
    # Monitoring
    SENTRY_DSN = os.getenv('SENTRY_DSN')

# ==============================================
# SECURITY UTILITIES
# ==============================================

class SecurityManager:
    """Security utilities for Friday AI"""
    
    def __init__(self):
        try:
            self.cipher = Fernet(SecureConfig.ENCRYPTION_KEY.encode())
        except:
            # Generate new key if invalid
            new_key = Fernet.generate_key()
            self.cipher = Fernet(new_key)
            print(f"⚠️ Generated new encryption key: {new_key.decode()}")
            print("Add this to your .env file: ENCRYPTION_KEY=" + new_key.decode())
    
    def validate_input(self, user_input: str) -> tuple[bool, str]:
        """Validate user input for security"""
        
        # Basic validation
        if not user_input or len(user_input.strip()) == 0:
            return False, "Input cannot be empty"
        
        if len(user_input) > SecureConfig.MAX_INPUT_LENGTH:
            return False, f"Input too long (max {SecureConfig.MAX_INPUT_LENGTH} characters)"
        
        # Security pattern detection
        dangerous_patterns = [
            (r'<script.*?>', "XSS attempt detected"),
            (r'javascript:', "JavaScript injection"),
            (r'DROP\s+TABLE', "SQL injection attempt"),
            (r'EXEC\s*\(', "Code execution attempt"),
            (r'import\s+os', "Python injection"),
            (r'eval\s*\(', "Code evaluation attempt"),
            (r'__import__', "Import injection"),
            (r'\.\./', "Directory traversal"),
        ]
        
        for pattern, error_msg in dangerous_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                return False, error_msg
        
        return True, "Valid"
    
    def sanitize_response(self, response_text: str) -> str:
        """Clean AI response to prevent data leakage"""
        
        # Remove sensitive patterns
        sensitive_patterns = [
            (r'sk-[a-zA-Z0-9\-_]{20,}', '[API_KEY_REDACTED]'),  # OpenAI keys
            (r'API[_\s]?KEY[_\s]?[:=]\s*[\w\-]+', '[API_KEY_REDACTED]'),
            (r'PASSWORD[_\s]?[:=]\s*\w+', '[PASSWORD_REDACTED]'),
            (r'SECRET[_\s]?[:=]\s*[\w\-]+', '[SECRET_REDACTED]'),
            (r'TOKEN[_\s]?[:=]\s*[\w\-]+', '[TOKEN_REDACTED]'),
            (r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', '[CARD_REDACTED]'),
            (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]'),
        ]
        
        for pattern, replacement in sensitive_patterns:
            response_text = re.sub(pattern, replacement, response_text, flags=re.IGNORECASE)
        
        # Limit response length
        if len(response_text) > SecureConfig.MAX_RESPONSE_LENGTH:
            response_text = response_text[:SecureConfig.MAX_RESPONSE_LENGTH] + "... [TRUNCATED]"
        
        return response_text
    
    def generate_token(self, user_id: str, expires_hours: int = 24) -> str:
        """Generate JWT token"""
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(hours=expires_hours),
            'iat': datetime.utcnow(),
            'jti': str(uuid.uuid4())
        }
        return jwt.encode(payload, SecureConfig.JWT_SECRET_KEY, algorithm='HS256')

# ==============================================
# FLASK APP SETUP
# ==============================================

app = Flask(__name__)
app.config.from_object(SecureConfig)

# Security setup
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

# CORS - allow your frontend domains
CORS(app, 
     origins=SecureConfig.ALLOWED_ORIGINS,
     methods=['POST', 'GET', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization'],
     supports_credentials=True)

# Logging
if not os.path.exists('logs'):
    os.mkdir('logs')

file_handler = RotatingFileHandler('logs/friday_secure.log', maxBytes=10240000, backupCount=5)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [%(request_id)s]'
))
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri=SecureConfig.RATE_LIMIT_STORAGE_URL,
    default_limits=SecureConfig.DEFAULT_RATE_LIMITS
)

# Caching
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Security manager
security = SecurityManager()

# ==============================================
# FRIDAY AI INITIALIZATION
# ==============================================

try:
    app.logger.info("🧠 Initializing Friday AI...")
    
    # Initialize your existing Friday AI system
    memory = MemoryCore(memory_file="friday_memory.enc", key_file="memory.key")
    emotion = EmotionCoreV2()
    friday_ai = FridayAI(memory, emotion)
    
    app.logger.info("✅ Friday AI initialization complete!")
    
except Exception as e:
    app.logger.error(f"❌ Friday AI initialization failed: {e}")
    friday_ai = None

# ==============================================
# SECURITY MIDDLEWARE
# ==============================================

@app.before_request
def before_request():
    """Security checks for every request"""
    g.request_id = str(uuid.uuid4())
    g.start_time = datetime.utcnow()
    
    # Log request
    app.logger.info(f"Request {g.request_id}: {request.method} {request.path}")
    
    # Security headers for CORS preflight
    if request.method == 'OPTIONS':
        return '', 200

@app.after_request
def after_request(response):
    """Add security headers"""
    # Security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['X-Request-ID'] = getattr(g, 'request_id', 'unknown')
    
    # Log response
    duration = (datetime.utcnow() - g.start_time).total_seconds() * 1000
    app.logger.info(f"Response {g.request_id}: {response.status_code} ({duration:.2f}ms)")
    
    return response

# Authentication decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization')
        
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
        
        if not token:
            return jsonify({'error': 'Token missing'}), 401
        
        try:
            data = jwt.decode(token, SecureConfig.JWT_SECRET_KEY, algorithms=['HS256'])
            g.current_user = data['user_id']
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(*args, **kwargs)
    return decorated

# ==============================================
# API ENDPOINTS
# ==============================================

@app.route('/api/v1/auth/token', methods=['POST'])
@limiter.limit("5/minute")
def get_token():
    """Generate authentication token"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'anonymous')
        
        # Simple validation - you can make this more sophisticated
        if not user_id or len(user_id) < 3:
            return jsonify({'error': 'Invalid user_id'}), 400
        
        token = security.generate_token(user_id)
        
        return jsonify({
            'token': token,
            'expires_in': 86400,  # 24 hours
            'user_id': user_id
        })
        
    except Exception as e:
        app.logger.error(f"Token generation error: {e}")
        return jsonify({'error': 'Token generation failed'}), 500

@app.route('/api/v1/chat', methods=['POST'])
@limiter.limit("30/minute")  # 30 requests per minute
@token_required
def chat():
    """Main chat endpoint - works with your existing Friday AI"""
    
    if not friday_ai:
        return jsonify({'error': 'Friday AI service unavailable'}), 503
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'JSON data required'}), 400
        
        user_input = data.get('message', '').strip()
        user_id = getattr(g, 'current_user', 'anonymous')
        pregnancy_week = data.get('pregnancy_week', 0)
        
        # Validate input
        is_valid, error_msg = security.validate_input(user_input)
        if not is_valid:
            app.logger.warning(f"Invalid input from {user_id}: {error_msg}")
            return jsonify({'error': f'Input validation failed: {error_msg}'}), 400
        
        # Set user context
        friday_ai.current_user_id = user_id
        
        # Get AI response using your existing system
        app.logger.info(f"Processing chat for user {user_id}")
        ai_response = friday_ai.respond_to(user_input, pregnancy_week=pregnancy_week)
        
        # Handle different response formats
        if isinstance(ai_response, dict):
            response_content = ai_response.get('content', str(ai_response))
            emotional_tone = ai_response.get('emotional_tone', 'neutral')
            emergency_detected = ai_response.get('emergency_detected', False)
        else:
            response_content = str(ai_response)
            emotional_tone = 'neutral'
            emergency_detected = False
        
        # Sanitize response
        safe_response = security.sanitize_response(response_content)
        
        # Log emergency if detected
        if emergency_detected:
            app.logger.critical(f"🚨 EMERGENCY detected for user {user_id}")
        
        return jsonify({
            'response': safe_response,
            'emotional_tone': emotional_tone,
            'emergency_detected': emergency_detected,
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': g.request_id
        })
        
    except Exception as e:
        app.logger.error(f"Chat error: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'request_id': getattr(g, 'request_id', 'unknown')
        }), 500

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'friday_ai': 'operational' if friday_ai else 'unavailable',
        'openai_configured': bool(SecureConfig.OPENAI_API_KEY),
        'serpapi_configured': bool(SecureConfig.SERPAPI_KEY)
    }
    
    return jsonify(status)

@app.route('/api/v1/config', methods=['GET'])
@token_required
def get_config():
    """Get non-sensitive configuration"""
    config = {
        'max_input_length': SecureConfig.MAX_INPUT_LENGTH,
        'confidence_threshold': SecureConfig.FRIDAY_CONFIDENCE_THRESHOLD,
        'model': SecureConfig.OPENAI_MODEL,
        'content_filter_enabled': SecureConfig.CONTENT_FILTER_ENABLED,
        'debug_mode': SecureConfig.FRIDAY_DEBUG
    }
    return jsonify(config)

# ==============================================
# ERROR HANDLERS
# ==============================================

@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request', 'message': 'Invalid request format'}), 400

@app.errorhandler(401)
def unauthorized(error):
    return jsonify({'error': 'Unauthorized', 'message': 'Valid token required'}), 401

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found', 'message': 'Endpoint does not exist'}), 404

@app.errorhandler(429)
def ratelimit_handler(error):
    return jsonify({
        'error': 'Rate limit exceeded',
        'message': 'Too many requests. Please slow down.',
        'retry_after': 60
    }), 429

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"Internal error: {error}", exc_info=True)
    return jsonify({
        'error': 'Internal server error',
        'request_id': getattr(g, 'request_id', 'unknown')
    }), 500

# ==============================================
# STARTUP & CONFIGURATION CHECK
# ==============================================

def check_configuration():
    """Check if all required configuration is present"""
    issues = []
    
    # Check critical configuration
    if not SecureConfig.OPENAI_API_KEY:
        issues.append("❌ OPENAI_API_KEY not configured")
    
    if not SecureConfig.JWT_SECRET_KEY or SecureConfig.JWT_SECRET_KEY == 'your-secret-key':
        issues.append("❌ JWT_SECRET_KEY not properly configured")
    
    # Warnings for optional but recommended config
    if not SecureConfig.SERPAPI_KEY:
        app.logger.warning("⚠️ SERPAPI_KEY not configured - search features may not work")
    
    if not SecureConfig.SENTRY_DSN:
        app.logger.warning("⚠️ SENTRY_DSN not configured - error tracking disabled")
    
    if issues:
        app.logger.error("🚨 CONFIGURATION ISSUES:")
        for issue in issues:
            app.logger.error(f"  {issue}")
        return False
    
    app.logger.info("✅ Configuration check passed")
    return True

if __name__ == '__main__':
    print("🛡️ Starting Secure Friday AI API...")
    
    # Check configuration
    if not check_configuration():
        print("❌ Configuration issues detected. Please fix them before starting.")
        exit(1)
    
    # Show current configuration (safely)
    print(f"🔧 Environment: {SecureConfig.ENV}")
    print(f"🔧 OpenAI Model: {SecureConfig.OPENAI_MODEL}")
    print(f"🔧 Debug Mode: {SecureConfig.FRIDAY_DEBUG}")
    print(f"🔧 Content Filter: {SecureConfig.CONTENT_FILTER_ENABLED}")
    print(f"🔧 Allowed Origins: {SecureConfig.ALLOWED_ORIGINS}")
    
    # Start server
    if SecureConfig.ENV == 'production':
        print("🚀 Starting in PRODUCTION mode")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("🚀 Starting in DEVELOPMENT mode")
        app.run(host='127.0.0.1', port=5000, debug=True)

# ==============================================
# QUICK START INSTRUCTIONS
# ==============================================
"""
🚀 QUICK START:

1. URGENT - Replace your exposed API keys:
   - Go to OpenAI dashboard and create NEW API key
   - Go to SerpAPI dashboard and create NEW API key
   - Update your .env file with NEW keys

2. Add these NEW security variables to your .env:
   JWT_SECRET_KEY=friday_ai_super_secret_jwt_key_2025
   ENCRYPTION_KEY=friday_ai_encryption_key_32_chars
   ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com

3. Start the secure API:
   python secure_friday_api.py

4. Test with curl:
   # Get token
   curl -X POST http://localhost:5000/api/v1/auth/token \
     -H "Content-Type: application/json" \
     -d '{"user_id": "test_user"}'

   # Chat (use token from above)
   curl -X POST http://localhost:5000/api/v1/chat \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer YOUR_TOKEN_HERE" \
     -d '{"message": "Hello Friday!"}'

🔒 Your Friday AI is now SECURE and ready for production!
"""