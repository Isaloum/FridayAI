# Contributing to FridayAI

First off, thank you for considering contributing to FridayAI! It's people like you that make FridayAI such a great tool.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inspiring community for all. Please be respectful and constructive in all interactions.

### Our Standards

- ✅ Be respectful and inclusive
- ✅ Accept constructive criticism gracefully
- ✅ Focus on what is best for the community
- ✅ Show empathy towards others

- ❌ Do not use inappropriate language or imagery
- ❌ Do not engage in trolling or insulting comments
- ❌ Do not harass others in any form
- ❌ Do not publish others' private information

## Getting Started

### Prerequisites

Before you begin, ensure you have:

- Python 3.11 or higher installed
- Git for version control
- A GitHub account
- Familiarity with Python and AI concepts

### Development Setup

1. **Fork and Clone**
   ```bash
   # Fork the repository on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/FridayAI.git
   cd FridayAI
   ```

2. **Set Up Remote**
   ```bash
   # Add the original repository as upstream
   git remote add upstream https://github.com/Isaloum/FridayAI.git
   ```

3. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov black pylint mypy  # Development tools
   ```

5. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your test credentials
   ```

6. **Verify Setup**
   ```bash
   pytest tests/ -v
   ```

## How to Contribute

### Types of Contributions

We welcome many types of contributions:

- 🐛 **Bug Reports**: Found a bug? Let us know!
- ✨ **Feature Requests**: Have an idea? Share it!
- 📝 **Documentation**: Improve our docs
- 🧪 **Tests**: Add or improve test coverage
- 🔧 **Bug Fixes**: Fix reported issues
- ⚡ **Features**: Implement new functionality
- 🎨 **UI/UX**: Improve user experience

### Contribution Workflow

1. **Create an Issue** (for significant changes)
   - Describe the problem or feature
   - Discuss the approach
   - Wait for maintainer feedback

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-number-description
   ```

3. **Make Your Changes**
   - Write clean, documented code
   - Follow coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

4. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "type: brief description
   
   Longer description if needed"
   ```
   
   Commit types:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `style:` Code style changes (formatting)
   - `refactor:` Code refactoring
   - `test:` Adding or updating tests
   - `chore:` Maintenance tasks

5. **Keep Your Branch Updated**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

6. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request**
   - Use a clear, descriptive title
   - Reference related issues
   - Describe what changed and why
   - Include screenshots for UI changes

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

```python
# Good example
class MemoryCore:
    """Manages memory storage and retrieval.
    
    Attributes:
        short_term: Short-term memory storage
        long_term: Long-term memory storage
    """
    
    def __init__(self, config: dict) -> None:
        """Initialize the memory core.
        
        Args:
            config: Configuration dictionary
        """
        self.short_term = []
        self.long_term = []
    
    def store_memory(self, memory: str, memory_type: str = "short") -> bool:
        """Store a memory in the appropriate storage.
        
        Args:
            memory: The memory content to store
            memory_type: Type of memory ("short" or "long")
            
        Returns:
            True if successful, False otherwise
        """
        if memory_type == "short":
            self.short_term.append(memory)
            return True
        return False
```

### Code Quality Tools

**Before committing, run:**

```bash
# Format code
black fridayai.py core/ tests/

# Check style
pylint fridayai.py core/ tests/

# Type checking
mypy fridayai.py core/ tests/

# Run tests
pytest tests/ -v --cov
```

### Code Quality Checklist

- [ ] Code follows PEP 8 style guide
- [ ] All functions have docstrings
- [ ] Type hints are used where appropriate
- [ ] No hardcoded credentials or secrets
- [ ] Error handling is implemented
- [ ] Code is DRY (Don't Repeat Yourself)
- [ ] Variable names are descriptive
- [ ] Complex logic has comments
- [ ] No debug print statements remain

## Testing Guidelines

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Use descriptive test function names

```python
# tests/test_memory_core.py
import pytest
from core.MemoryCore import MemoryCore

class TestMemoryCore:
    """Test suite for MemoryCore functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.memory_core = MemoryCore(config={})
    
    def test_store_short_term_memory(self):
        """Test storing a memory in short-term storage."""
        result = self.memory_core.store_memory("Test memory", "short")
        assert result is True
        assert len(self.memory_core.short_term) == 1
    
    def test_store_invalid_memory_type(self):
        """Test handling of invalid memory type."""
        result = self.memory_core.store_memory("Test", "invalid")
        assert result is False
```

### Test Coverage

- Aim for 80%+ code coverage
- Test both success and failure cases
- Test edge cases and boundary conditions
- Mock external dependencies (API calls, database)

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_memory_core.py -v

# Run with coverage report
pytest tests/ --cov=. --cov-report=html

# Run only fast tests (skip slow integration tests)
pytest tests/ -v -m "not slow"
```

## Pull Request Process

### Before Submitting

1. ✅ Update documentation if needed
2. ✅ Add tests for new features
3. ✅ Ensure all tests pass
4. ✅ Update CHANGELOG.md if applicable
5. ✅ Run code quality tools
6. ✅ Rebase on latest main branch

### PR Template

```markdown
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Code refactoring
- [ ] Performance improvement

## Related Issue
Fixes #(issue number)

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
- [ ] All existing tests pass
- [ ] Added new tests
- [ ] Manual testing completed

## Screenshots (if applicable)
[Add screenshots here]

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests added/updated
```

### Review Process

1. Maintainers will review your PR
2. Address any feedback or requested changes
3. Once approved, your PR will be merged
4. Your contribution will be credited!

## Reporting Bugs

### Before Reporting

- Check if the bug is already reported in [Issues](https://github.com/Isaloum/FridayAI/issues)
- Try to reproduce the bug consistently
- Gather relevant information (logs, screenshots, etc.)

### Bug Report Template

```markdown
**Describe the Bug**
Clear and concise description

**To Reproduce**
Steps to reproduce:
1. Go to '...'
2. Click on '...'
3. See error

**Expected Behavior**
What you expected to happen

**Screenshots**
If applicable, add screenshots

**Environment:**
- OS: [e.g., Windows 10]
- Python Version: [e.g., 3.11.2]
- FridayAI Version: [e.g., 1.0.0]

**Additional Context**
Any other relevant information
```

## Suggesting Enhancements

### Enhancement Template

```markdown
**Is your feature request related to a problem?**
Clear description of the problem

**Describe the solution you'd like**
What you want to happen

**Describe alternatives you've considered**
Alternative solutions or features

**Additional context**
Any other context or screenshots
```

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Credited in release notes
- Given a shout-out on social media (if desired)

## Questions?

- Open a [Discussion](https://github.com/Isaloum/FridayAI/discussions)
- Check existing documentation
- Ask in the community

---

**Thank you for contributing to FridayAI! 🎉**
