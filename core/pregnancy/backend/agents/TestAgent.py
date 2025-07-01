class TestAgent:
    def should_handle(self, message):
        return "test" in message.lower()
        
    def process(self, request_data):
        return {
            "content": "✅ Test Agent is working!",
            "agent": "TestAgent",
            "received_data": request_data
        }