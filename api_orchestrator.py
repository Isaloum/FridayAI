# ==============================================
# File: C:\Users\ihabs\FridayAI\api_orchestrator.py (Modernized)
# Purpose: Production-ready Friday AI API orchestrator
# ==============================================

class APIOrchestrator:

    # Auth
    def register(request):
        user = AuthManager.create_user(request.body)
        token = AuthManager.issue_jwt(user.id)
        return { "token": token, "user": user }

    def login(request):
        user = AuthManager.verify(request.body)
        token = AuthManager.issue_jwt(user.id)
        return { "token": token, "user": user }

    # Chat
    def chat(request):
        user = AuthManager.check(request.headers.token)
        context = MemoryContextInjector.load(user.id)
        mood = EmotionClassifier.detect(request.body.message)
        if PregnancySupportCore.applicable(request.body):
            reply = PregnancySupportCore.handle(request.body, context)
        else:
            reply = ChatCore.generate(request.body, context, mood)
        MemoryCore.store(user.id, request.body.message, reply)
        return { "response": reply, "detectedMood": mood }

    # Memory
    def get_memory(request):
        user = AuthManager.check(request.headers.token)
        return MemoryCore.retrieve(user.id, limit=request.query.limit)

    # Upload
    def upload_config(request):
        user = AuthManager.check(request.headers.token)
        return UploadConfig.save(request.files.file)

    def upload_knowledge(request):
        user = AuthManager.check(request.headers.token)
        return KnowledgeUploader.save(request.files.file)

    # Pregnancy Profile
    def get_preg_profile(request):
        user = AuthManager.check(request.headers.token)
        return PregnancyBackend.get_profile(user.id)

    def update_preg_profile(request):
        user = AuthManager.check(request.headers.token)
        return PregnancyBackend.update_profile(user.id, request.body)

    def weekly_update(request):
        user = AuthManager.check(request.headers.token)
        return PregnancyBackend.weekly_update(user.id, week=request.query.week)
