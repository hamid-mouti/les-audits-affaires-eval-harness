# Changelog

All notable changes to the Les Audits-Affaires Evaluation Harness will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-06-22

### Added
- Initial release of Les Audits-Affaires Evaluation Harness
- Complete evaluation framework for French legal LLM benchmark
- 5-category evaluation system (action_requise, delai_legal, documents_obligatoires, impact_financier, consequences_non_conformite)
- Azure OpenAI GPT-4o integration as expert evaluator
- Asynchronous batch processing with concurrency control
- Multiple output formats (JSON, CSV, Excel, Markdown)
- Progress tracking and intermediate result saving
- Robust error handling with retry mechanisms
- Visualization tools for score distributions and correlations
- Comprehensive analysis reporting
- Web service endpoint for HTTP API evaluation
- French README as primary documentation
- English README as secondary documentation
- Proper Python package structure with setup.py
- Unit tests and comprehensive setup verification
- CLI interface with argparse
- Environment configuration with .env support
- Chat template formatting for model prompting
- Results directory management
- Makefile for project management

### Features
- **Dataset Integration**: Direct integration with `legmlai/les-audits-affaires` dataset
- **Model Evaluation**: Support for any HTTP endpoint following the specified format
- **Expert Evaluation**: Azure OpenAI GPT-4o as legal expert evaluator
- **Scalable Processing**: Asynchronous processing with configurable concurrency
- **Rich Analytics**: Statistical analysis and visualization tools
- **Flexible Configuration**: Environment-based configuration system
- **Professional Structure**: Proper Python package with src/ layout
- **Multi-language Documentation**: French and English documentation
- **Service Integration**: HTTP API for integration with other systems
- **Testing Framework**: Comprehensive test suite and setup verification

### Requirements
- Python 3.8+
- Azure OpenAI API access
- Model endpoint (HTTP API)
- 18 Python dependencies (see requirements.txt)

### Supported Formats
- Input: HTTP POST JSON requests
- Output: JSON, CSV, Excel, Markdown reports
- Chat Template: Custom format with reasoning tokens
- Model Response: Multiple format support (text, content, message, response)

## Future Versions

### Planned for 1.1.0
- [ ] Advanced filtering and sampling options
- [ ] Model comparison features
- [ ] Enhanced visualization dashboard
- [ ] Additional evaluator models support
- [ ] Benchmarking against other legal datasets
- [ ] Docker container support
- [ ] CI/CD pipeline integration 