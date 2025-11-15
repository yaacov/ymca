# Model Converter Documentation

The YMCA Model Converter downloads models from Hugging Face and converts them to GGUF format for use with llama.cpp.

## Overview

The model converter provides:
- **Download**: Fetch models from Hugging Face Hub with authentication support
- **Conversion**: Convert PyTorch/SafeTensors models to GGUF format
- **Quantization**: Apply various quantization levels for smaller model sizes
- **Optimization**: Choose output types (F16, F32, Q8_0) for different use cases

## Quick Start

### Basic Conversion

Download and convert a model:

```bash
ymca-convert meta-llama/Llama-3.2-1B-Instruct
```

### With Quantization

Convert and quantize to 4-bit:

```bash
ymca-convert meta-llama/Llama-3.2-1B-Instruct --quantize q4_k_m
```

### Download Only

Download without converting:

```bash
ymca-convert meta-llama/Llama-3.2-1B-Instruct --download-only
```

## Installation

The converter is included with YMCA.

### Setup Virtual Environment

Create and activate a virtual environment:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Install YMCA

With the virtual environment activated:

```bash
pip install -e .
```

### Python Dependencies

These are installed automatically:
- `transformers` - Model downloading and metadata
- `huggingface-hub` - Hugging Face API access
- `torch` - PyTorch model loading
- `safetensors` - SafeTensors format support
- `gguf` - GGUF format support

### llama.cpp Requirement

**Important:** For model conversion and quantization, you need the **llama.cpp** repository (separate from `llama-cpp-python`).

**The Good News:** YMCA will try to find and use llama.cpp automatically. It searches for:
- `./llama.cpp` (current directory)
- `../llama.cpp` (parent directory)
- `~/llama.cpp` (home directory)
- `/usr/local/llama.cpp` (system location)

If llama.cpp is not found, YMCA will create instruction files showing you how to install it manually.

### Installing llama.cpp (if needed)

If the converter can't find llama.cpp, install it:

```bash
# Clone llama.cpp repository
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build the conversion and quantization tools
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release

# Return to your project directory
cd ../..
```

YMCA will automatically detect the llama.cpp directory and use it for conversions.

**Note:** You only need llama.cpp for **converting models**. Running models only requires `llama-cpp-python` (installed with YMCA).

## Command-Line Options

### Basic Usage

```bash
ymca-convert MODEL_NAME [OPTIONS]
```

### Model Selection

```bash
ymca-convert MODEL_NAME              # Model name from Hugging Face
ymca-convert org/model-name          # Full path with organization
ymca-convert meta-llama/Llama-3.2-1B-Instruct  # Example
```

### Output Directory

```bash
ymca-convert MODEL_NAME --output-dir DIR
```

Default: `models/` in current directory

### Download Options

```bash
ymca-convert MODEL_NAME --download-only      # Skip conversion
ymca-convert MODEL_NAME --force-download     # Re-download existing models
ymca-convert MODEL_NAME --token TOKEN        # HF authentication token
```

### Conversion Options

```bash
ymca-convert MODEL_NAME --output-type TYPE   # F32, F16, or Q8_0 (default: F16)
ymca-convert MODEL_NAME --quantize QUANT     # Quantization method
```

### Logging

```bash
ymca-convert MODEL_NAME --verbose            # Detailed logging
```

## Quantization Methods

Quantization reduces model size and memory usage while maintaining accuracy.

### Available Methods

| Method | Bits | Size Reduction | Quality | Speed | Use Case |
|--------|------|----------------|---------|-------|----------|
| f16 | 16 | None (baseline) | Highest | Slowest | Reference/testing |
| f32 | 32 | None | Highest | Slowest | Full precision |
| q8_0 | 8 | 50% | Excellent | Fast | High quality, reasonable size |
| q6_k | 6 | 60% | Very Good | Fast | Balanced |
| q5_k_m | 5 | 65% | Good | Faster | Recommended balance |
| q5_k_s | 5 | 65% | Good | Faster | Slightly smaller than q5_k_m |
| q4_k_m | 4 | 70% | Good | Very Fast | Best for most uses |
| q4_k_s | 4 | 70% | Acceptable | Very Fast | Smaller, slight quality drop |
| q4_0 | 4 | 70% | Acceptable | Very Fast | Legacy format |
| q3_k_m | 3 | 75% | Fair | Fastest | Aggressive compression |
| q2_k | 2 | 85% | Poor | Fastest | Experimental |

### Recommended Quantization

**Best Overall Quality (q5_k_m)**
```bash
ymca-convert model-name --quantize q5_k_m
```
Excellent balance of size, speed, and quality.

**Best for Limited RAM (q4_k_m)**
```bash
ymca-convert model-name --quantize q4_k_m
```
Good quality with significant size reduction.

**Maximum Quality (q8_0)**
```bash
ymca-convert model-name --quantize q8_0
```
Minimal quality loss, still 50% smaller than f16.

**Aggressive Compression (q3_k_m)**
```bash
ymca-convert model-name --quantize q3_k_m
```
For very limited hardware, acceptable quality loss.

### K-Quants vs Legacy

**K-Quants (Recommended)**: q4_k_m, q5_k_m, q6_k
- Better quality at same bit-width
- Mixed precision strategies
- Optimal for modern use

**Legacy**: q4_0, q5_0
- Older format
- Simpler quantization
- Use K-quants instead

## Output Types

Control the initial GGUF conversion format:

### F16 (Default)
```bash
ymca-convert model-name --output-type F16
```
- Half precision (16-bit floats)
- Good balance for most models
- Recommended starting point

### F32
```bash
ymca-convert model-name --output-type F32
```
- Full precision (32-bit floats)
- Highest quality, largest size
- Use for maximum accuracy

### Q8_0
```bash
ymca-convert model-name --output-type Q8_0
```
- 8-bit quantization
- Direct conversion without F16 intermediate
- Faster for immediate quantized output

## Authentication

### Hugging Face Token

For private or gated models:

```bash
# Via command line
ymca-convert model-name --token YOUR_HF_TOKEN

# Via environment variable
export HF_TOKEN=YOUR_HF_TOKEN
ymca-convert model-name
```

### Getting a Token

1. Go to https://huggingface.co/settings/tokens
2. Create a new token with read access
3. Copy and use in conversion command

## Workflow

### Step-by-Step Conversion

1. **Download** from Hugging Face:
   ```bash
   ymca-convert meta-llama/Llama-3.2-1B-Instruct --download-only
   ```

2. **Convert** to GGUF (F16):
   ```bash
   ymca-convert meta-llama/Llama-3.2-1B-Instruct
   ```

3. **Quantize** to desired level:
   ```bash
   ymca-convert meta-llama/Llama-3.2-1B-Instruct --quantize q4_k_m
   ```

### All-in-One

Download, convert, and quantize in one command:

```bash
ymca-convert meta-llama/Llama-3.2-1B-Instruct --quantize q4_k_m
```

## Output Structure

Models are organized by name:

```
models/
└── meta-llama_llama-3.2-1b-instruct/
    ├── model/                    # Original downloaded model
    │   ├── config.json
    │   ├── model.safetensors
    │   └── tokenizer.json
    └── gguf/                     # Converted GGUF files
        ├── meta-llama_llama-3.2-1b-instruct-f16.gguf
        ├── meta-llama_llama-3.2-1b-instruct-q4_k_m.gguf
        └── meta-llama_llama-3.2-1b-instruct-q5_k_m.gguf
```

## Examples

### Convert IBM Granite Model

```bash
ymca-convert ibm-granite/granite-4.0-h-tiny --quantize q4_k_m
```

### Convert with Custom Output Directory

```bash
ymca-convert meta-llama/Llama-3.2-1B-Instruct \
  --output-dir /path/to/models \
  --quantize q5_k_m
```

### Convert Multiple Quantizations

```bash
# Q4 for speed
ymca-convert model-name --quantize q4_k_m

# Q5 for balance
ymca-convert model-name --quantize q5_k_m

# Q8 for quality
ymca-convert model-name --quantize q8_0
```
