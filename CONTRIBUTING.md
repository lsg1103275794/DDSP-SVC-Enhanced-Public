# Contributing to DDSP-SVC-Enhanced

Thank you for your interest in contributing to DDSP-SVC-Enhanced! This document provides guidelines for contributing to the project.

## 🌟 Ways to Contribute

- 🐛 **Bug Reports**: Report bugs via [GitHub Issues](https://github.com/yourusername/DDSP-SVC-Enhanced/issues)
- 💡 **Feature Requests**: Suggest new features or enhancements
- 📝 **Documentation**: Improve or translate documentation
- 🔧 **Code Contributions**: Submit bug fixes or new features
- 🎨 **UI/UX Improvements**: Enhance the web interface
- 🧪 **Testing**: Test new features and report results

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/your-username/DDSP-SVC-Enhanced.git
cd DDSP-SVC-Enhanced

# Add upstream remote
git remote add upstream https://github.com/original/DDSP-SVC-Enhanced.git
```

### 2. Create a Branch

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/bug-description
```

### 3. Set Up Development Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install pytest black flake8 mypy

# Set up pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

## 📋 Coding Guidelines

### Python Code Style

- Follow **PEP 8** style guide
- Use **4 spaces** for indentation (no tabs)
- Maximum line length: **100 characters**
- Use **type hints** where applicable

```python
# Good example
def process_audio(
    audio: np.ndarray,
    sample_rate: int,
    hop_size: int = 512
) -> torch.Tensor:
    """Process audio with DDSP.

    Args:
        audio: Input audio array
        sample_rate: Sampling rate in Hz
        hop_size: Hop size for STFT

    Returns:
        Processed audio tensor
    """
    # Implementation...
    pass
```

### Code Formatting

Use **Black** for automatic formatting:

```bash
black ddsp/ api/ --line-length 100
```

### Linting

Run **flake8** before committing:

```bash
flake8 ddsp/ api/ --max-line-length 100 --ignore E203,W503
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_ddsp.py

# Run with coverage
pytest --cov=ddsp --cov-report=html
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Use descriptive test names

```python
def test_f0_smoother_reduces_jitter():
    """Test that F0 smoothing reduces pitch jitter."""
    # Arrange
    f0 = torch.randn(1, 100, 1) * 10 + 440

    # Act
    smoothed_f0 = apply_smoothing(f0)

    # Assert
    assert torch.std(smoothed_f0) < torch.std(f0)
```

## 📝 Commit Guidelines

### Commit Message Format

Follow **Conventional Commits** specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**

```bash
# Good commit messages
git commit -m "feat(lfo): add tremolo depth parameter"
git commit -m "fix(api): resolve CORS issue in web GUI"
git commit -m "docs(readme): add installation instructions for macOS"

# With body
git commit -m "feat(effects): add phaser effect to audio chain

- Implement all-pass filter cascade
- Add modulation control parameters
- Update effects chain configuration"
```

### Commit Best Practices

- ✅ Write clear, descriptive commit messages
- ✅ Keep commits atomic (one logical change per commit)
- ✅ Test your changes before committing
- ❌ Don't commit commented-out code
- ❌ Don't commit large files or model weights
- ❌ Don't commit sensitive information

## 🔀 Pull Request Process

### Before Submitting

1. **Update from upstream:**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests:**
   ```bash
   pytest
   ```

3. **Check code style:**
   ```bash
   black --check ddsp/ api/
   flake8 ddsp/ api/
   ```

4. **Update documentation** if needed

### Submitting PR

1. **Push your branch:**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request** on GitHub

3. **Fill out PR template:**
   - Description of changes
   - Related issue numbers
   - Testing performed
   - Screenshots (for UI changes)

### PR Requirements

- ✅ All tests pass
- ✅ Code follows style guidelines
- ✅ Documentation updated (if applicable)
- ✅ No merge conflicts
- ✅ Descriptive PR title and description

### Review Process

- Maintainers will review your PR
- Address review comments promptly
- Be respectful and constructive
- PRs may require changes before merging

## 🎯 Priority Areas

We especially welcome contributions in these areas:

### High Priority
- 🐛 **Bug Fixes**: Critical bugs affecting core functionality
- 📊 **Performance**: Optimization of audio processing
- 🌐 **Web GUI**: UI/UX improvements
- 📱 **Mobile Support**: Responsive design enhancements

### Medium Priority
- 🎵 **Audio Effects**: New effect implementations
- 📚 **Documentation**: Tutorials and guides
- 🌍 **Internationalization**: Translations
- 🧪 **Testing**: Increase test coverage

### Low Priority
- ♻️ **Refactoring**: Code cleanup (must not break existing features)
- 🎨 **UI Polish**: Visual enhancements
- 📈 **Analytics**: Usage statistics and monitoring

## 🏗️ Project Structure

```
DDSP-SVC-Enhanced/
├── ddsp/                  # Core DDSP modules
│   ├── vocoder.py        # DDSP vocoder
│   ├── enhancements.py   # Audio enhancements
│   ├── lfo.py           # LFO modulation
│   ├── effects_chain.py # Effects processing
│   └── ...
├── api/                  # FastAPI backend
│   ├── routes/          # API endpoints
│   ├── services/        # Business logic
│   └── models/          # Data schemas
├── web/                  # Vue.js frontend
│   └── src/
│       ├── views/       # Page components
│       └── api/         # API client
├── reflow/              # Rectified Flow model
├── encoder/             # Audio encoders
├── configs/             # Configuration files
├── docs/                # Documentation
└── tests/               # Test suite
```

## 📚 Additional Resources

### Documentation
- [Training Guide](docs/Training_Guide.md)
- [API Documentation](docs/API_Documentation.md)
- [Enhancement Technical Guide](docs/AudioNoise_Technical_Analysis.md)

### Related Projects
- [Original DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)
- [AudioNoise](https://github.com/torvalds/AudioNoise)

## 💬 Communication

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Pull Requests**: For code contributions

## 📜 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors.

### Expected Behavior

- ✅ Be respectful and considerate
- ✅ Welcome newcomers and help them learn
- ✅ Accept constructive criticism gracefully
- ✅ Focus on what's best for the project

### Unacceptable Behavior

- ❌ Harassment or discrimination
- ❌ Trolling or insulting comments
- ❌ Personal or political attacks
- ❌ Publishing others' private information

## ⚖️ Legal

### Contributor License

By contributing to this project, you agree that:

1. You have the right to submit the contribution
2. Your contribution is licensed under the project's MIT License
3. You grant the project maintainers a perpetual license to use your contribution

### Attribution

All contributors will be recognized in:
- GitHub Contributors page
- Project README (for significant contributions)
- Release notes (for major features)

## 🙏 Thank You!

Thank you for contributing to DDSP-SVC-Enhanced! Your contributions help make voice conversion technology more accessible and powerful for everyone.

---

**Questions?** Open an issue or discussion on GitHub!
