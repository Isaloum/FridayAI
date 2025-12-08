# FridayAI 🤖🧠

> An advanced AI assistant with cognitive architecture, emotional intelligence, and comprehensive memory systems

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

## 🌟 Overview

FridayAI is a sophisticated artificial intelligence system designed with human-like cognitive capabilities. It features multi-layered reasoning, emotional intelligence, long-term memory, and specialized knowledge domains including pregnancy care support.

### Key Features

- 🧠 **Advanced Cognitive Architecture**
  - Multi-layer reasoning and decision-making
  - Intent detection and routing
  - Goal planning and execution
  - Self-awareness and reflection mechanisms

- ❤️ **Emotional Intelligence**
  - Real-time mood detection and management
  - Empathy-driven responses
  - Emotional memory integration
  - Tone adaptation based on context

- 💾 **Sophisticated Memory Systems**
  - Short-term conversational memory
  - Long-term episodic memory
  - Graph-based knowledge storage
  - Vector-based semantic search
  - Memory reflection and consolidation

- 🤰 **Specialized Pregnancy Support**
  - Maternal care companion
  - Pregnancy tracking and guidance
  - Emotional support during pregnancy
  - Medical information assistance

- 🔧 **Additional Capabilities**
  - Multi-user support with personas
  - Web search integration
  - Tool execution framework
  - API service interface
  - Natural language understanding

## 🚀 Quick Start

### Prerequisites

- **Python 3.11 or higher**
- **pip** (Python package manager)
- **OpenAI API Key** (for GPT models)
- PostgreSQL (optional, for production database)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Isaloum/FridayAI.git
   cd FridayAI
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env with your actual credentials
   # At minimum, add your OPENAI_API_KEY
   ```

5. **Run FridayAI**
   ```bash
   # Command-line interface
   python fridayai.py
   
   # Or start the API server
   python run.py
   ```

### First Interaction

Once running, you can interact with Friday:

```
You: Hello Friday, how are you today?
Friday: Hello! I'm doing well, thank you for asking. How can I assist you today?
```

## 📁 Project Structure

```
FridayAI/
├── core/                      # Core cognitive modules
│   ├── EmotionCore.py        # Emotion detection and management
│   ├── MemoryCore.py         # Memory systems
│   ├── DialogueCore.py       # Conversation management
│   ├── PlanningCore.py       # Goal planning
│   └── ...                   # Other core modules
│
├── tests/                     # Test suite
│   ├── test_emotion_core.py
│   ├── test_api_service.py
│   └── ...
│
├── tools/                     # Utility tools
├── memory/                    # Memory storage
├── docs/                      # Documentation and research papers
├── utils/                     # Helper utilities
│
├── fridayai.py               # Main application entry point
├── api_service.py            # Flask API service
├── run.py                    # Server launcher
├── requirements.txt          # Python dependencies
├── .env.example             # Environment template
└── README.md                # This file
```

## 🎯 Usage Examples

### Using the Command Line Interface

```python
# Interactive conversation
python fridayai.py

# With specific input
python fridayai.py --input "What's the weather like?"
```

### Using the API Service

```bash
# Start the API server
python run.py
```

```python
# Make a request
import requests

response = requests.post('http://localhost:5050/api/chat', json={
    'message': 'Tell me about your capabilities',
    'context': {}
})

print(response.json())
```

### Using as a Python Module

```python
from fridayai import FridayAI

# Initialize Friday
friday = FridayAI()

# Get a response
response = friday.process_input("Hello, Friday!")
print(response)
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_emotion_core.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## 📚 Documentation

- **[Setup Guide](docs/SETUP.md)** - Detailed installation and configuration
- **[API Documentation](docs/API.md)** - REST API endpoints and usage
- **[Architecture Overview](docs/ARCHITECTURE.md)** - System design and components
- **[Security Guidelines](SECURITY_AUDIT.md)** - Security best practices
- **[Contributing](CONTRIBUTING.md)** - How to contribute to the project

## 🏗️ Architecture

FridayAI uses a modular, layered architecture:

```
┌─────────────────────────────────────────────────┐
│           User Interface Layer                   │
│     (CLI, API, Web Interface)                    │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│         Cognitive Processing Layer               │
│  (Intent Detection, Reasoning, Planning)         │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│          Emotional Intelligence Layer            │
│   (Mood Detection, Empathy, Tone Adaptation)    │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│            Memory Management Layer               │
│ (STM, LTM, Knowledge Graph, Vector Search)      │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│              Foundation Layer                    │
│       (LLM Integration, Database, Tools)         │
└─────────────────────────────────────────────────┘
```

## 🔒 Security

Security is a top priority for FridayAI. Please follow these guidelines:

- ⚠️ **Never commit API keys or credentials** to version control
- 🔑 Use environment variables (`.env` file) for all secrets
- 🛡️ Keep your OpenAI API key secure and rotate it regularly
- 🔐 Use strong passwords for database access
- 📋 Review `SECURITY_AUDIT.md` for detailed security guidelines

### Reporting Security Issues

If you discover a security vulnerability, please email [your-email@example.com] instead of using the issue tracker.

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Make your changes**
4. **Run tests** to ensure nothing breaks
5. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
6. **Push to the branch** (`git push origin feature/AmazingFeature`)
7. **Open a Pull Request**

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and development process.

## 🗺️ Roadmap

### Current Version (v1.0)
- ✅ Core cognitive architecture
- ✅ Emotional intelligence
- ✅ Memory systems
- ✅ Pregnancy support module
- ✅ API service

### Upcoming Features (v1.1)
- 🔄 Enhanced multi-modal input (voice, images)
- 🔄 Improved web search integration
- 🔄 Advanced planning capabilities
- 🔄 Plugin system for extensions
- 🔄 Web-based user interface

### Future Vision (v2.0)
- 🎯 Autonomous goal achievement
- 🎯 Multi-agent collaboration
- 🎯 Continuous learning system
- 🎯 Mobile application
- 🎯 Cloud deployment options

See [MULTI-PROJECT ROADMAP.txt](MULTI-PROJECT%20ROADMAP.txt) for detailed roadmap.

## 📊 Project Statistics

- **Lines of Code**: ~30,000+
- **Python Modules**: 318
- **Core Components**: 50+
- **Test Coverage**: Growing
- **Active Development**: Yes

## 🌐 Community

- **GitHub Issues**: [Report bugs or request features](https://github.com/Isaloum/FridayAI/issues)
- **Discussions**: [Join the conversation](https://github.com/Isaloum/FridayAI/discussions)
- **Documentation**: [Read the docs](docs/)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT models
- LangChain community for NLP tools
- All contributors and users of FridayAI

## 📧 Contact

- **Project Lead**: Isaloum
- **GitHub**: [@Isaloum](https://github.com/Isaloum)
- **Repository**: [https://github.com/Isaloum/FridayAI](https://github.com/Isaloum/FridayAI)

---

<div align="center">

**Built with ❤️ and 🧠 by the FridayAI Team**

[⭐ Star us on GitHub](https://github.com/Isaloum/FridayAI) | [🐛 Report Bug](https://github.com/Isaloum/FridayAI/issues) | [💡 Request Feature](https://github.com/Isaloum/FridayAI/issues)

</div>
