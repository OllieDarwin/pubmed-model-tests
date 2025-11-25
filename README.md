# Biomedical LLM Evaluation Suite

A comprehensive evaluation framework for testing language models on biomedical literature interpretation tasks. This project evaluates models on their ability to analyze agent-pathway relationships in scientific abstracts through four key tasks: relevance assessment, mechanism extraction, evidence quality evaluation, and parsing stability.

## Overview

This evaluation suite uses a **two-stage pipeline** to test models that may struggle with structured output:

1. **Generation Stage**: The evaluation model (e.g., Galactica, BioMistral) generates plaintext analysis
2. **Parsing Stage**: Instructor + a parser model (local or API-based) extracts structured data from the plaintext

This approach enables evaluation of biomedical models without requiring native JSON formatting capabilities.

## Key Features

- **Four Evaluation Tasks**: Relevance, mechanism extraction, evidence quality, and parsing stability
- **Flexible Architecture**: Support for both causal and seq2seq models
- **Local or Cloud Parsing**: Use Ollama models locally or OpenAI API for parsing
- **Type-Safe Outputs**: Pydantic models ensure structured, validated results
- **Comprehensive Metrics**: Accuracy, within-1-step scoring, component extraction rates
- **Visual Analytics**: Performance visualizations and detailed reports

## Quick Start

### Prerequisites

- Python 3.9+
- HuggingFace account (for gated models)
- (Optional) Ollama installed for local parsing

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd pubmed-model-tests

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Login to HuggingFace
huggingface-cli login
```

### Local Parsing Setup (Recommended)

For local, free parsing with Ollama:

```bash
# Install Ollama from https://ollama.ai

# Pull a parsing model
ollama pull llama3.2
# or
ollama pull mistral
```

### Running the Evaluation

1. Open [model_evaluation.ipynb](model_evaluation.ipynb) in Jupyter or Google Colab

2. Configure your evaluation in the Configuration cell:

```python
# Model to evaluate (generates plaintext responses)
MODEL_NAME = 'facebook/galactica-1.3b'

# Parser provider for structured extraction
PARSER_PROVIDER = "ollama/llama3.2"  # or "ollama/mistral", "openai/gpt-4o-mini"
```

3. Run all cells to execute the full evaluation suite

## Architecture

### Core Components

#### ModelEvaluator
Loads and manages HuggingFace models for text generation:
- Automatic device detection (GPU/CPU)
- Support for both causal LM and seq2seq models
- Configurable generation parameters
- Automatic tokenizer configuration

#### InstructorParser
Extracts structured data from plaintext using Instructor:
- Multiple provider support (Ollama, OpenAI)
- Type-safe Pydantic model outputs
- Graceful error handling with sensible defaults

#### Pydantic Models
- `RelevanceResult`: Binary classification with rationale
- `MechanismResult`: Mechanism summary, components, and direction
- `QualityResult`: Evidence quality on 4-point scale

### Test Suite

Located in [model_tests/](model_tests/), each JSON file contains abstracts with gold-standard labels:

| Test File | Purpose | Evaluation Metric |
|-----------|---------|-------------------|
| [test_relevance.json](model_tests/test_relevance.json) | Does abstract explain agent-pathway relationship? | Exact match accuracy |
| [test_mechanism.json](model_tests/test_mechanism.json) | Extract mechanism direction (activation/inhibition) | Exact match accuracy |
| [test_quality.json](model_tests/test_quality.json) | Assess evidence strength (4 levels) | Exact match + within-1-step |
| [test_stability.json](model_tests/test_stability.json) | Parsing consistency across duplicate runs | Consistency rate |

### Test Item Structure

Each test item includes:
```json
{
  "id": "unique_identifier",
  "agent": "Drug or compound name",
  "pathway": "Biological pathway",
  "abstract": "Scientific abstract text",
  "gold_label": "Expected answer"
}
```

## Evaluation Process

### 1. Relevance Assessment
**Task**: Determine if an abstract explains HOW an agent affects a pathway through a molecular mechanism.

**Strict Criteria**:
- Must contain direct molecular interactions OR explicit mechanistic implications
- NOT relevant if only outcomes or associations are mentioned

**Metric**: Exact match accuracy

### 2. Mechanism Extraction
**Task**: Extract the direction of effect (activation/inhibition) and molecular components.

**Extracted Data**:
- One-sentence mechanism summary
- List of molecular components (genes, proteins, molecules)
- Direction: activation, inhibition, or unknown

**Metrics**:
- Direction accuracy
- Component extraction rate

### 3. Evidence Quality
**Task**: Assess research evidence strength on a 4-point scale.

**Quality Levels**:
- **Strong**: Multiple experimental approaches + clinical/in vivo validation
- **Moderate**: Solid experimental support but lacks clinical validation
- **Weak**: Minimal experimental data
- **Insufficient**: Unclear methods or very limited evidence

**Metrics**:
- Exact match accuracy
- Within-1-step accuracy (adjacent quality levels)

### 4. Parsing Stability
**Task**: Verify parsing consistency by running duplicate evaluations.

**Method**: Run identical prompts twice and compare outputs

**Metric**: Consistency rate across runs

## Output and Results

The evaluation generates:

1. **Per-Test DataFrames**: Detailed results for each evaluation task
2. **Summary Statistics**: Accuracy, timing, and component extraction metrics
3. **Visualizations**:
   - Performance by test type (bar chart)
   - Generation time distribution (histogram)
4. **Exportable Summary**: CSV-ready summary DataFrame

### Example Output

```
================================================================================
EVALUATION SUMMARY
================================================================================
Evaluation Model:  facebook/galactica-1.3b
Parser Provider:   ollama/llama3.2
================================================================================
Test 1 - Relevance Accuracy:    85.0%
Test 2 - Mechanism Accuracy:    75.0%
Test 3 - Evidence Quality:      60.0%
Test 4 - Parsing Stability:     90.0%
Avg Generation Time:            3.2s per item
Avg Parse Time:                 1.5s per item
================================================================================
```

## Supported Models

### Evaluation Models
Any HuggingFace model supporting causal LM or seq2seq:
- `facebook/galactica-*` (biomedical focus)
- `microsoft/BioMistral-*` (biomedical focus)
- `meta-llama/Llama-*`
- `mistralai/Mistral-*`
- And many more...

### Parser Providers
- **Ollama (Local)**:
  - `ollama/llama3.2` (recommended)
  - `ollama/mistral`
  - Any other Ollama model
- **OpenAI (API)**:
  - `openai/gpt-4o-mini`
  - `openai/gpt-4o`
  - Requires `OPENAI_API_KEY` environment variable

## Configuration Options

### Model Configuration
```python
evaluator = ModelEvaluator(
    model_name='facebook/galactica-1.3b',
    device='auto'  # 'auto', 'cuda', or 'cpu'
)
```

### Generation Parameters
Modify in `ModelEvaluator.generate_response()`:
- `max_new_tokens`: Maximum tokens to generate (default: 512)
- `temperature`: Sampling temperature (default: 0.7)
- `top_p`: Nucleus sampling parameter (default: 0.9)
- `repetition_penalty`: Reduce repetition (default: 1.1)

### Parser Configuration
```python
parser = InstructorParser(
    provider="ollama/llama3.2"  # or other provider
)
```

## Performance Considerations

### GPU vs CPU
- **GPU (CUDA)**: 10-50x faster generation, recommended for larger models
- **CPU**: Works but slower, suitable for small models (<1B parameters)

### Memory Requirements
- **1-2B models**: ~4-8GB RAM/VRAM
- **7B models**: ~14-28GB RAM/VRAM
- **13B+ models**: 26GB+ RAM/VRAM

### Speed Optimization
- Use smaller models for faster evaluation
- Run on GPU when available
- Use local Ollama parsing to avoid API latency
- Reduce `max_new_tokens` for shorter responses

## Troubleshooting

### Common Issues

**HuggingFace Authentication Error**
```bash
huggingface-cli login
# Enter your token from https://huggingface.co/settings/tokens
```

**Ollama Connection Error**
```bash
# Ensure Ollama is running
ollama serve

# Verify model is pulled
ollama list
ollama pull llama3.2
```

**CUDA Out of Memory**
- Reduce model size or switch to CPU
- Reduce `max_new_tokens`
- Close other GPU-intensive applications

**Parser Errors**
- Check provider string format
- For OpenAI, verify `OPENAI_API_KEY` is set
- For Ollama, ensure service is running

## Extending the Framework

### Adding New Tests
1. Create a new JSON file in [model_tests/](model_tests/) following the existing structure
2. Define a new Pydantic model if needed
3. Create a prompt template
4. Add evaluation loop in the notebook

### Custom Evaluation Metrics
Modify the results dictionaries in each test section to include additional metrics.

### Supporting New Model Types
The `ModelEvaluator` class automatically handles causal LM and seq2seq models. For other architectures, extend the `__init__` method.

## Project Structure

```
pubmed-model-tests/
├── model_evaluation.ipynb    # Main evaluation notebook
├── requirements.txt           # Python dependencies
├── CLAUDE.md                 # Project instructions for Claude Code
├── README.md                 # This file
└── model_tests/              # Test data directory
    ├── test_relevance.json   # Relevance assessment tests
    ├── test_mechanism.json   # Mechanism extraction tests
    ├── test_quality.json     # Evidence quality tests
    └── test_stability.json   # Parsing stability tests
```

## Dependencies

### Core Libraries
- `torch>=2.0.0`: PyTorch for model inference
- `transformers>=4.35.0`: HuggingFace Transformers
- `instructor>=1.0.0`: Structured output extraction
- `pydantic>=2.0.0`: Type-safe data models

### Utilities
- `pandas>=2.0.0`: Data manipulation
- `matplotlib>=3.7.0`, `seaborn>=0.12.0`: Visualization
- `tqdm>=4.65.0`: Progress bars

### Optional
- `sentencepiece>=0.1.99`: For certain tokenizers
- `accelerate>=0.24.0`: For efficient model loading

## Acknowledgments

- Built with [Instructor](https://github.com/jxnl/instructor) for structured extraction
- Uses [HuggingFace Transformers](https://huggingface.co/transformers/) for model loading
- Ollama integration for local, free parsing
