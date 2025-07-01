# ==============================================
# File: C:\Users\ihabs\FridayAI\api_service.py (Production)
# Purpose: Production-ready Friday AI API service
# ==============================================

from flask import Flask, request, jsonify, send_from_directory
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from datetime import datetime
from functools import wraps
from logging.handlers import RotatingFileHandler
import logging, os, uuid
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

# Optional imports
try:
    from healthcheck import HealthCheck
except ImportError:
    HealthCheck = None

try:
    from flask_swagger_ui import get_swaggerui_blueprint
except ImportError:
    get_swaggerui_blueprint = None

# Core AI and services
try:
    from FridayAI import FridayAI
except ImportError:
    from fridayai import FridayAI
from MemoryCore import MemoryCore
from EmotionCore import EmotionCore

# Configuration
class Config:
    ENV = os.getenv('FLASK_ENV', 'production')
    SENTRY_DSN = os.getenv('SENTRY_DSN')
    CACHE_TYPE = 'redis' if ENV == 'production' else 'simple'
    RATE_LIMIT = "1000/day;100/hour"
    SWAGGER_URL = '/docs'
    OPENAPI_JSON = '/openapi.json'

# App init
app = Flask(__name__)
app.config.from_object(Config)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1)
CORS(app)
limiter = Limiter(key_func=get_remote_address, default_limits=[Config.RATE_LIMIT], app=app)
cache = Cache(app)
if HealthCheck:
    HealthCheck(app, "/healthcheck")

# Observability
if Config.SENTRY_DSN:
    sentry_sdk.init(
        dsn=Config.SENTRY_DSN,
        integrations=[FlaskIntegration()],
        traces_sample_rate=1.0,
        environment=Config.ENV
    )

# Logging
handler = RotatingFileHandler('friday_api.log', maxBytes=1_000_000, backupCount=5)
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# Core AI Initialization
memory = MemoryCore(memory_file="friday_memory.enc", key_file="memory.key")
emotion = EmotionCore()
ai = FridayAI(memory, emotion)

# Swagger UI
if get_swaggerui_blueprint:
    swagger_bp = get_swaggerui_blueprint(
        Config.SWAGGER_URL,
        Config.OPENAPI_JSON,
        config={'app_name': 'Friday AI API'}
    )
    app.register_blueprint(swagger_bp, url_prefix=Config.SWAGGER_URL)

@app.route(Config.OPENAPI_JSON)
def openapi_spec():
    return send_from_directory(os.getcwd(), 'openapi.json')

# Middleware
@app.before_request
def before_request():
    request.start_time = datetime.now()
    request.id = str(uuid.uuid4())
    app.logger.info(f"{request.method} {request.path} (ID: {request.id})")

@app.after_request
def after_request(response):
    duration = (datetime.now() - request.start_time).total_seconds() * 1000
    app.logger.info(f"{response.status_code} in {duration:.2f}ms (ID: {request.id})")
    response.headers['X-Request-ID'] = request.id
    return response

# Helpers
def json_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not request.is_json:
            return jsonify(error="Request must be JSON"), 400
        return f(*args, **kwargs)
    return wrapper

# Endpoints
@app.route("/api/v1/respond", methods=["POST"])
@limiter.limit("60/minute")
@json_required
@cache.cached(timeout=10, query_string=True)
def respond():
    data = request.json
    user_input = data.get('input', '').strip()
    if not user_input:
        return jsonify(error="Input cannot be empty"), 400

    if data.get('domain'):
        ai.domain_adapter.load_domain(data['domain'], context=data.get('context', {}))

    result = ai.respond_to(user_input)
    if result.get('priority') == 'high':
        app.logger.warning('High priority response')

    return jsonify(
        response=result['content'],
        emotional_tone=result.get('emotional_tone'),
        suggestions=result.get('suggestions', []),
        metadata={
            'request_id': request.id,
            'domain': data.get('domain'),
            'timestamp': datetime.now().isoformat()
        }
    )

@app.route("/api/v1/reflect", methods=["POST"])
@json_required
def reflect():
    try:
        res = ai.reflection_loop.run_reflection_cycle(
            self_awareness=ai.self_awareness,
            belief_updater=ai.belief_updater
        )
        return jsonify(
            status="success",
            insights=res.get('insights', []),
            belief_updates=res.get('belief_updates', 0)
        )
    except Exception as e:
        app.logger.error(f"Reflection error: {e}")
        return jsonify(error="Reflection failed"), 500

@app.route("/api/v1/dream", methods=["POST"])
@limiter.limit("10/hour")
def dream():
    try:
        d = ai.narrative_fusion.simulate_internal_event('idle_dream')
        return jsonify(dream=d, timestamp=datetime.now().isoformat())
    except Exception as e:
        app.logger.error(f"Dream error: {e}")
        return jsonify(error="Dream generation failed"), 500

# Error handlers
@app.errorhandler(429)
def ratelimit(e): return jsonify(error="Rate limit exceeded", message=str(e)), 429
@app.errorhandler(404)
def notfound(e): return jsonify(error="Not found"), 404
@app.errorhandler(500)
def internalerr(e): return jsonify(error="Server error"), 500

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=(Config.ENV == 'development')
    )
