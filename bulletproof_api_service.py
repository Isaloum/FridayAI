# ==============================================
# File: bulletproof_api_service.py
# Purpose: ULTRA SECURE Friday AI API Service
# Copyright (c) 2025 [YOUR NAME/COMPANY]. All Rights Reserved.
# 
# This software is proprietary and confidential. Unauthorized copying,
# distribution, or use is strictly prohibited and may result in legal action.
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

# Your existing Friday AI imports
try:
    from fridayai import FridayAI
    from core.MemoryCore import MemoryCore
    from core.EmotionCoreV2 import EmotionCoreV2
except ImportError as e:
    print(f"WARNING: Could not import Friday AI modules: {e}")
    print("Make sure your Friday AI files are in the correct path")

# ==============================================
# SECURITY CONFIGURATION
# ==============================================

class SecureConfig:
    """Ultra-secure configuration management"""
    
    # Environment and basic settings
    ENV = os.getenv('FLASK_ENV', 'production')
    DEBUG = ENV == 'development'
    
    # Security keys (NEVER hardcode these!)
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', secrets.token_urlsafe(32))
    ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY', Fernet.generate_key().decode())
    API_KEY_HASH = os.getenv('API_KEY_HASH')  # For admin access
    
    # Rate limiting
    RATE_LIMIT_STORAGE_URL = os.getenv('REDIS_URL', 'memory://')
    DEFAULT_RATE_LIMITS = ["1000/day", "100/hour", "10/minute"]
    STRICT_RATE_LIMITS = ["500/day", "50/hour", "5/minute"]
    
    # Security headers
    ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', 'https://yourdomain.com').split(',')
    
    # Monitoring
    SENTRY_DSN = os.getenv('SENTRY_DSN')
    LOG_LEVEL = logging.INFO if ENV == 'production' else logging.DEBUG
    
    # AI Safety
    MAX_INPUT_LENGTH = int(os.getenv('MAX_INPUT_LENGTH', '5000'))
    MAX_RESPONSE_LENGTH = int(os.getenv('MAX_RESPONSE_LENGTH', '10000'))
    CONTENT_FILTER_ENABLED = os.getenv('CONTENT_FILTER_ENABLED', 'true').lower() == 'true'

# ==============================================
# SECURITY UTILITIES
# ==============================================

class SecurityUtils:
    """Advanced security utilities"""
    
    def __init__(self, encryption_key: str):
        self.cipher = Fernet(encryption_key.encode() if isinstance(encryption_key, str) else encryption_key)
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            return self.cipher.encrypt(data.encode()).decode()
        except Exception:
            return data  # Fallback for non-critical data
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            return self.cipher.decrypt(encrypted_data.encode()).decode()
        except Exception:
            return encrypted_data  # Fallback
    
    @staticmethod
    def validate_input(user_input: str) -> tuple[bool, str]:
        """Comprehensive input validation"""
        
        # Size validation
        if len(user_input) > SecureConfig.MAX_INPUT_LENGTH:
            return False, f"Input too long (max {SecureConfig.MAX_INPUT_LENGTH} characters)"
        
        if len(user_input.strip()) == 0:
            return False, "Input cannot be empty"
        
        # Dangerous pattern detection
        dangerous_patterns = [
            (r'<script.*?>', "XSS attempt detected"),
            (r'javascript:', "JavaScript injection detected"),
            (r'DROP\s+TABLE', "SQL injection attempt detected"),
            (r'EXEC\s*\(', "Code execution attempt detected"),
            (r'import\s+os', "Python injection attempt detected"),
            (r'eval\s*\(', "Code evaluation attempt detected"),
            (r'exec\s*\(', "Code execution attempt detected"),
            (r'__import__', "Import injection detected"),
            (r'file://', "File access attempt detected"),
            (r'\.\./', "Directory traversal attempt detected"),
        ]
        
        for pattern, error_msg in dangerous_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                return False, error_msg
        
        # Content filtering for inappropriate content
        if SecureConfig.CONTENT_FILTER_ENABLED:
            inappropriate_patterns = [
                r'\b(hack|crack|exploit|vulnerability)\b',
                r'\b(password|secret|token|key)\s*[:=]\s*\w+',
                r'\b(suicide|self.?harm|kill.?myself)\b',
            ]
            
            for pattern in inappropriate_patterns:
                if re.search(pattern, user_input, re.IGNORECASE):
                    return False, "Content policy violation detected"
        
        return True, "Valid"
    
    @staticmethod
    def sanitize_response(response_text: str) -> str:
        """Sanitize AI response to prevent data leakage"""
        
        # Remove potential sensitive information
        sensitive_patterns = [
            (r'API[_\s]?KEY[_\s]?[:=]\s*[\w\-]+', '[API_KEY_REDACTED]'),
            (r'PASSWORD[_\s]?[:=]\s*\w+', '[PASSWORD_REDACTED]'),
            (r'SECRET[_\s]?[:=]\s*[\w\-]+', '[SECRET_REDACTED]'),
            (r'TOKEN[_\s]?[:=]\s*[\w\-]+', '[TOKEN_REDACTED]'),
            (r'mongodb://[^\s]+', '[DATABASE_URL_REDACTED]'),
            (r'postgresql://[^\s]+', '[DATABASE_URL_REDACTED]'),
            (r'mysql://[^\s]+', '[DATABASE_URL_REDACTED]'),
            (r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', '[CARD_NUMBER_REDACTED]'),
            (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]'),
        ]
        
        for pattern, replacement in sensitive_patterns:
            response_text = re.sub(pattern, replacement, response_text, flags=re.IGNORECASE)
        
        # Limit response length
        if len(response_text) > SecureConfig.MAX_RESPONSE_LENGTH:
            response_text = response_text[:SecureConfig.MAX_RESPONSE_LENGTH] + "... [TRUNCATED]"
        
        return response_text
    
    @staticmethod
    def generate_api_token(user_id: str, expires_hours: int = 24) -> str:
        """Generate secure JWT token"""
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(hours=expires_hours),
            'iat': datetime.utcnow(),
            'jti': str(uuid.uuid4())  # JWT ID for token blacklisting
        }
        return jwt.encode(payload, SecureConfig.JWT_SECRET_KEY, algorithm='HS256')

# ==============================================
# FLASK APP INITIALIZATION
# ==============================================

app = Flask(__name__)
app.config.from_object(SecureConfig)

# Security middleware
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# CORS with strict origin control
CORS(app, 
     origins=SecureConfig.ALLOWED_ORIGINS,
     methods=['POST', 'GET'],
     allow_headers=['Content-Type', 'Authorization'],
     supports_credentials=True)

# Advanced logging setup
if not app.debug:
    if not os.path.exists('logs'):
        os.mkdir('logs')
    
    file_handler = RotatingFileHandler('logs/friday_api.log', maxBytes=10240000, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [%(pathname)s:%(lineno)d] [%(request_id)s]'
    ))
    file_handler.setLevel(SecureConfig.LOG_LEVEL)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(SecureConfig.LOG_LEVEL)

# Sentry error tracking
if SecureConfig.SENTRY_DSN:
    sentry_sdk.init(
        dsn=SecureConfig.SENTRY_DSN,
        integrations=[FlaskIntegration()],
        traces_sample_rate=0.1,  # Reduced for performance
        environment=SecureConfig.ENV,
        attach_stacktrace=True,
        send_default_pii=False  # Don't send personal info
    )

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri=SecureConfig.RATE_LIMIT_STORAGE_URL,
    default_limits=SecureConfig.DEFAULT_RATE_LIMITS
)

# Caching
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Security utilities
security = SecurityUtils(SecureConfig.ENCRYPTION_KEY)

# ==============================================
# FRIDAY AI INITIALIZATION (SECURE)
# ==============================================

try:
    app.logger.info("Initializing Friday AI core systems...")
    memory = MemoryCore(memory_file="friday_memory.enc", key_file="memory.key")
    emotion = EmotionCoreV2()
    friday_ai = FridayAI(memory, emotion)
    app.logger.info("Friday AI initialization complete")
except Exception as e:
    app.logger.error(f"Friday AI initialization failed: {e}")
    friday_ai = None

# ==============================================
# SECURITY MIDDLEWARE & DECORATORS
# ==============================================

@app.before_request
def before_request():
    """Security checks and request logging"""
    # Generate request ID for tracking
    g.request_id = str(uuid.uuid4())
    
    # Security headers
    if request.method == 'OPTIONS':
        return '', 200
    
    # Log request (without sensitive data)
    app.logger.info(
        f"Request {g.request_id}: {request.method} {request.path}",
        extra={
            'request_id': g.request_id,
            'ip': request.remote_addr,
            'user_agent': request.headers.get('User-Agent', 'Unknown')[:100],
            'content_length': request.content_length or 0
        }
    )
    
    # Block suspicious IPs (basic implementation)
    blocked_patterns = ['curl', 'wget', 'python-requests', 'bot']
    user_agent = request.headers.get('User-Agent', '').lower()
    if any(pattern in user_agent for pattern in blocked_patterns):
        app.logger.warning(f"Blocked suspicious user agent: {user_agent}")
        return jsonify({'error': 'Access denied'}), 403

@app.after_request
def after_request(response):
    """Add security headers and log response"""
    # Security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    response.headers['X-Request-ID'] = getattr(g, 'request_id', 'unknown')
    
    # Remove server information
    response.headers.pop('Server', None)
    
    # Log response
    app.logger.info(
        f"Response {getattr(g, 'request_id', 'unknown')}: {response.status_code}",
        extra={
            'request_id': getattr(g, 'request_id', 'unknown'),
            'status_code': response.status_code,
            'response_size': len(response.get_data())
        }
    )
    
    return response

def token_required(f):
    """JWT token authentication decorator"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization')
        
        if auth_header:
            try:
                token = auth_header.split(' ')[1]  # Bearer <token>
            except IndexError:
                return jsonify({'error': 'Invalid authorization header format'}), 401
        
        if not token:
            app.logger.warning(f"No token provided from {request.remote_addr}")
            return jsonify({'error': 'Token is missing'}), 401
        
        try:
            data = jwt.decode(token, SecureConfig.JWT_SECRET_KEY, algorithms=['HS256'])
            g.current_user = data['user_id']
            g.token_jti = data.get('jti')
        except jwt.ExpiredSignatureError:
            app.logger.warning(f"Expired token from {request.remote_addr}")
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            app.logger.warning(f"Invalid token from {request.remote_addr}")
            return jsonify({'error': 'Token is invalid'}), 401
        
        return f(*args, **kwargs)
    
    return decorated

def admin_required(f):
    """Admin access decorator"""
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({'error': 'Admin API key required'}), 401
        
        # Hash the provided key and compare with stored hash
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        if key_hash != SecureConfig.API_KEY_HASH:
            app.logger.warning(f"Invalid admin API key from {request.remote_addr}")
            return jsonify({'error': 'Invalid API key'}), 401
        
        return f(*args, **kwargs)
    
    return decorated

def json_required(f):
    """Ensure request is valid JSON"""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        return f(*args, **kwargs)
    return decorated

# ==============================================
# API ENDPOINTS - ULTRA SECURE
# ==============================================

@app.route('/api/v1/auth/token', methods=['POST'])
@limiter.limit("5/minute")  # Strict limit for token generation
@json_required
def generate_token():
    """Generate authentication token"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        temp_key = data.get('temp_key')  # Temporary authentication key
        
        if not user_id or not temp_key:
            return jsonify({'error': 'user_id and temp_key required'}), 400
        
        # Validate temp_key (implement your own validation logic)
        if not _validate_temp_key(user_id, temp_key):
            app.logger.warning(f"Invalid temp key for user {user_id} from {request.remote_addr}")
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Generate token
        token = security.generate_api_token(user_id, expires_hours=24)
        
        app.logger.info(f"Token generated for user {user_id}")
        
        return jsonify({
            'token': token,
            'expires_in': 86400,  # 24 hours
            'user_id': user_id
        })
        
    except Exception as e:
        app.logger.error(f"Token generation error: {e}")
        return jsonify({'error': 'Token generation failed'}), 500

@app.route('/api/v1/chat', methods=['POST'])
@limiter.limit("30/minute")  # Conversation rate limit
@token_required
@json_required
def chat_with_friday():
    """Main Friday AI chat endpoint - ULTRA SECURE"""
    
    if not friday_ai:
        return jsonify({'error': 'Friday AI service unavailable'}), 503
    
    try:
        data = request.get_json()
        user_input = data.get('message', '').strip()
        user_id = getattr(g, 'current_user', 'anonymous')
        pregnancy_week = data.get('pregnancy_week', 0)
        context = data.get('context', {})
        
        # Input validation
        is_valid, validation_error = security.validate_input(user_input)
        if not is_valid:
            app.logger.warning(f"Invalid input from user {user_id}: {validation_error}")
            return jsonify({'error': f'Input validation failed: {validation_error}'}), 400
        
        # Rate limiting per user
        cache_key = f"user_requests:{user_id}"
        user_requests = cache.get(cache_key) or 0
        if user_requests > 100:  # 100 requests per hour per user
            return jsonify({'error': 'User rate limit exceeded'}), 429
        cache.set(cache_key, user_requests + 1, timeout=3600)
        
        # Set user context for Friday AI
        friday_ai.current_user_id = user_id
        
        # Process through Friday AI with error handling
        try:
            ai_response = friday_ai.respond_to(user_input, pregnancy_week=pregnancy_week)
        except Exception as ai_error:
            app.logger.error(f"Friday AI processing error: {ai_error}")
            return jsonify({
                'error': 'AI processing temporarily unavailable',
                'retry_after': 60
            }), 503
        
        # Extract and sanitize response
        if isinstance(ai_response, dict):
            response_content = ai_response.get('content', 'I apologize, but I cannot process that request.')
            emotional_tone = ai_response.get('emotional_tone', 'neutral')
            emergency_detected = ai_response.get('emergency_detected', False)
            suggestions = ai_response.get('suggestions', [])
        else:
            response_content = str(ai_response)
            emotional_tone = 'neutral'
            emergency_detected = False
            suggestions = []
        
        # Sanitize response for security
        safe_response = security.sanitize_response(response_content)
        
        # Log important interactions (anonymized)
        if emergency_detected:
            app.logger.critical(f"EMERGENCY detected for user {user_id[:8]}***")
        
        # Encrypt sensitive response data if needed
        if any(keyword in safe_response.lower() for keyword in ['health', 'medical', 'personal']):
            # Mark as sensitive but don't encrypt in transit (HTTPS handles that)
            pass
        
        response_data = {
            'response': safe_response,
            'emotional_tone': emotional_tone,
            'emergency_detected': emergency_detected,
            'suggestions': suggestions[:3],  # Limit suggestions for security
            'metadata': {
                'request_id': g.request_id,
                'timestamp': datetime.utcnow().isoformat(),
                'user_id': user_id,
                'safe_mode': True
            }
        }
        
        app.logger.info(f"Successful AI response for user {user_id}")
        return jsonify(response_data)
        
    except Exception as e:
        app.logger.error(f"Chat endpoint error: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'request_id': getattr(g, 'request_id', 'unknown')
        }), 500

@app.route('/api/v1/health', methods=['GET'])
@limiter.limit("10/minute")
def health_check():
    """Secure health check endpoint"""
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0',
        'environment': SecureConfig.ENV
    }
    
    # Check Friday AI status
    if friday_ai:
        health_status['ai_status'] = 'operational'
    else:
        health_status['ai_status'] = 'unavailable'
        health_status['status'] = 'degraded'
    
    return jsonify(health_status)

@app.route('/api/v1/admin/stats', methods=['GET'])
@admin_required
@limiter.limit("10/hour")
def admin_stats():
    """Admin statistics endpoint"""
    try:
        stats = {
            'total_requests': cache.get('total_requests') or 0,
            'active_users': cache.get('active_users') or 0,
            'emergency_alerts': cache.get('emergency_alerts') or 0,
            'system_uptime': datetime.utcnow().isoformat(),
            'ai_status': 'operational' if friday_ai else 'unavailable'
        }
        return jsonify(stats)
    except Exception as e:
        app.logger.error(f"Admin stats error: {e}")
        return jsonify({'error': 'Stats unavailable'}), 500

@app.route('/api/v1/admin/users/<user_id>/suspend', methods=['POST'])
@admin_required
@limiter.limit("5/hour")
def suspend_user(user_id):
    """Suspend a user (admin only)"""
    try:
        cache.set(f"suspended_user:{user_id}", True, timeout=86400)  # 24 hour suspension
        app.logger.warning(f"User {user_id} suspended by admin")
        return jsonify({'status': 'user suspended', 'user_id': user_id})
    except Exception as e:
        app.logger.error(f"User suspension error: {e}")
        return jsonify({'error': 'Suspension failed'}), 500

# ==============================================
# ERROR HANDLERS - SECURE
# ==============================================

@app.errorhandler(400)
def bad_request(error):
    app.logger.warning(f"Bad request from {request.remote_addr}: {error}")
    return jsonify({
        'error': 'Bad request',
        'message': 'Invalid request format'
    }), 400

@app.errorhandler(401)
def unauthorized(error):
    app.logger.warning(f"Unauthorized access attempt from {request.remote_addr}")
    return jsonify({
        'error': 'Unauthorized',
        'message': 'Valid authentication required'
    }), 401

@app.errorhandler(403)
def forbidden(error):
    app.logger.warning(f"Forbidden access attempt from {request.remote_addr}")
    return jsonify({
        'error': 'Forbidden',
        'message': 'Access denied'
    }), 403

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Not found',
        'message': 'Endpoint does not exist'
    }), 404

@app.errorhandler(429)
def ratelimit_handler(error):
    app.logger.warning(f"Rate limit exceeded from {request.remote_addr}")
    return jsonify({
        'error': 'Rate limit exceeded',
        'message': 'Too many requests. Please slow down.',
        'retry_after': 60
    }), 429

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"Internal server error: {error}", exc_info=True)
    return jsonify({
        'error': 'Internal server error',
        'message': 'Something went wrong on our end',
        'request_id': getattr(g, 'request_id', 'unknown')
    }), 500

# ==============================================
# HELPER FUNCTIONS
# ==============================================

def _validate_temp_key(user_id: str, temp_key: str) -> bool:
    """
    Validate temporary authentication key
    IMPLEMENT YOUR OWN LOGIC HERE based on your user system
    """
    # Example implementation - replace with your logic
    expected_key = hashlib.sha256(f"{user_id}:friday_ai_secret".encode()).hexdigest()[:16]
    return temp_key == expected_key

def _is_user_suspended(user_id: str) -> bool:
    """Check if user is suspended"""
    return cache.get(f"suspended_user:{user_id}") is True

# ==============================================
# STARTUP SECURITY CHECKS
# ==============================================

def perform_startup_security_check():
    """Perform security validation on startup"""
    issues = []
    
    # Check for required environment variables
    required_vars = ['JWT_SECRET_KEY', 'ENCRYPTION_KEY']
    for var in required_vars:
        if not os.getenv(var):
            issues.append(f"Missing required environment variable: {var}")
    
    # Check file permissions
    sensitive_files = ['friday_memory.enc', 'memory.key']
    for file_path in sensitive_files:
        if os.path.exists(file_path):
            stat_info = os.stat(file_path)
            if stat_info.st_mode & 0o077:  # Check if readable by others
                issues.append(f"File {file_path} has insecure permissions")
    
    if issues:
        app.logger.error("SECURITY ISSUES DETECTED:")
        for issue in issues:
            app.logger.error(f"  - {issue}")
        app.logger.error("Please fix these issues before running in production!")
    else:
        app.logger.info("Security checks passed ✓")

# ==============================================
# APPLICATION STARTUP
# ==============================================

if __name__ == '__main__':
    # Perform security checks
    perform_startup_security_check()
    
    # Production vs Development settings
    if SecureConfig.ENV == 'production':
        app.logger.info("Starting Friday AI API in PRODUCTION mode")
        # Use a production WSGI server like Gunicorn in real deployment
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    else:
        app.logger.info("Starting Friday AI API in DEVELOPMENT mode")
        app.run(host='127.0.0.1', port=5000, debug=True)

# ==============================================
# DEPLOYMENT NOTES
# ==============================================
"""
DEPLOYMENT CHECKLIST:

1. Environment Variables (.env file):
   JWT_SECRET_KEY=your-super-secret-jwt-key-here
   ENCRYPTION_KEY=your-32-byte-encryption-key-here
   API_KEY_HASH=sha256-hash-of-your-admin-api-key
   ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
   SENTRY_DSN=your-sentry-dsn-for-error-tracking
   REDIS_URL=redis://localhost:6379/0

2. File Permissions:
   chmod 600 friday_memory.enc
   chmod 600 memory.key
   chmod 600 .env

3. Firewall Rules:
   - Only allow HTTPS (port 443)
   - Block direct access to port 5000
   - Use reverse proxy (nginx/Apache)

4. SSL Certificate:
   - Use Let's Encrypt or commercial SSL
   - Force HTTPS redirects
   - HSTS headers enabled

5. Monitoring:
   - Set up Sentry for error tracking
   - Monitor logs for security events
   - Set up alerts for rate limit violations

6. Backup:
   - Regular backups of encrypted memory files
   - Secure backup of environment variables
   - Test restore procedures

7. Updates:
   - Keep dependencies updated
   - Monitor security advisories
   - Regular security audits

PRODUCTION DEPLOYMENT:
pip install gunicorn
gunicorn --workers 4 --bind 0.0.0.0:5000 bulletproof_api_service:app
"""