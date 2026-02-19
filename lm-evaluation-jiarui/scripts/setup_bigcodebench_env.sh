#!/bin/bash
# Setup BigCodeBench evaluation environment
# Faithful to the official Docker/Evaluate.Dockerfile from https://github.com/bigcode-project/bigcodebench
#
# Usage: ./setup_bigcodebench_env.sh
#
# This installs all dependencies required for local BigCodeBench evaluation
# without Docker, using the current conda environment.

set -e

echo "=============================================="
echo "Setting up BigCodeBench Evaluation Environment"
echo "=============================================="
echo ""

# Check if we're in a conda environment
if [ -z "$CONDA_PREFIX" ]; then
    echo "WARNING: No conda environment detected. Installing to user site-packages."
    PIP_CMD="pip install --user"
else
    echo "Using conda environment: $CONDA_PREFIX"
    PIP_CMD="pip install"
fi

echo ""
echo "Step 1: Installing core bigcodebench package..."
echo "------------------------------------------------"
$PIP_CMD bigcodebench==0.2.2.dev2

echo ""
echo "Step 2: Installing evaluation dependencies (from requirements-eval.txt)..."
echo "--------------------------------------------------------------------------"
# These are the exact versions from the official BigCodeBench repository
# https://github.com/bigcode-project/bigcodebench/blob/main/Requirements/requirements-eval.txt

$PIP_CMD \
    beautifulsoup4==4.8.2 \
    blake3==0.4.1 \
    chardet==5.2.0 \
    cryptography==38.0.0 \
    Django==4.2.7 \
    dnspython==2.6.1 \
    docxtpl==0.11.5 \
    Faker==20.1.0 \
    flask_login==0.6.3 \
    flask_restful==0.3.10 \
    flask_wtf==1.2.1 \
    Flask-Mail==0.9.1 \
    flask==3.0.3 \
    folium==0.16.0 \
    gensim==4.3.2 \
    geopandas==0.13.2 \
    geopy==2.4.1 \
    holidays==0.29 \
    Levenshtein==0.25.0 \
    lxml==4.9.3 \
    matplotlib==3.7.0 \
    mechanize==0.4.9 \
    natsort==7.1.1 \
    networkx==2.6.3 \
    nltk==3.8 \
    openpyxl==3.1.2 \
    pandas==2.0.3 \
    Pillow==10.3.0 \
    prettytable==3.10.0 \
    psutil==5.9.5 \
    pycryptodome==3.14.1 \
    pyfakefs==5.4.1 \
    pyquery==1.4.3 \
    pytest==8.2.0 \
    python_http_client==3.3.7 \
    python-dateutil==2.9.0 \
    python-docx==1.1.0 \
    pytz==2023.3.post1 \
    PyYAML==6.0.1 \
    requests_mock==1.11.0 \
    requests==2.31.0 \
    rsa==4.9 \
    scikit-learn==1.3.1 \
    seaborn==0.13.2 \
    sendgrid==6.11.0 \
    shapely==2.0.4 \
    soundfile==0.12.1 \
    statsmodels==0.14.0 \
    sympy==1.12 \
    textblob==0.18.0 \
    texttable==1.7.0 \
    Werkzeug==3.0.1 \
    wikipedia==1.4.0 \
    wordcloud==1.9.3 \
    wordninja==2.0.0 \
    WTForms==3.1.2 \
    xlrd==2.0.1 \
    xlwt==1.3.0 \
    xmltodict==0.13.0

echo ""
echo "Step 3: Installing additional packages (may have version conflicts, installing without version pins)..."
echo "------------------------------------------------------------------------------------------------------"
# These packages have complex dependencies, install without strict version pins
$PIP_CMD \
    librosa \
    opencv-python-headless \
    scipy \
    scikit-image \
    selenium \
    pytesseract

echo ""
echo "Step 4: Installing optional heavy packages (keras/tensorflow - skip if not needed)..."
echo "-------------------------------------------------------------------------------------"
# Uncomment if you need keras/tensorflow support (large packages)
# $PIP_CMD keras==2.11.0 tensorflow==2.11.0

echo ""
echo "Step 5: Downloading NLTK data..."
echo "--------------------------------"
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True); nltk.download('wordnet', quiet=True)" 2>/dev/null || true

echo ""
echo "Step 6: Pre-downloading BigCodeBench datasets..."
echo "-------------------------------------------------"
python -c "
from datasets import load_dataset
print('Downloading bigcode/bigcodebench (full)...')
load_dataset('bigcode/bigcodebench', split='v0.1.4')
print('Downloading bigcode/bigcodebench-hard...')
load_dataset('bigcode/bigcodebench-hard', split='v0.1.4')
print('Done!')
" 2>/dev/null || echo "Dataset download skipped (may already exist)"

echo ""
echo "=============================================="
echo "BigCodeBench Environment Setup Complete!"
echo "=============================================="
echo ""
echo "To run local evaluation:"
echo "  python run_bigcodebench_local_eval.py --samples <file.jsonl>"
echo ""
echo "Or using the bigcodebench CLI:"
echo "  bigcodebench.evaluate --split instruct --subset full --samples <file.jsonl>"
echo ""
