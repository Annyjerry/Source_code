# Source Code Severity Analyzer

An AI-powered security tool that classifies the vulnerability severity of source code using a hybrid approach of **Deep Learning (CodeBERT)** and **Statistical NLP (TF-IDF)**. 



## Overview
This project provides a web-based interface built with **Streamlit** that allows developers to paste source code snippets (Python, Java, C, etc.) and receive an instant security assessment. The system categorizes code into four severity levels: **Low, Medium, High, and Critical**, providing detailed descriptions and actionable fixes for each.

## Model Architecture
The underlying engine uses a **Feature Fusion** technique:
1.  **CodeBERT:** Captures semantic meaning and contextual logic of the code.
2.  **TF-IDF:** Captures syntactic keywords and rare vulnerable function signatures.
3.  **Neural Network (MLP):** A multi-layer perceptron classifier trained on the concatenated features to predict the final severity label.



## Installation & Setup

### Prerequisites
- Python 3.9 or higher
- Windows Users: [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) installed.

### 1. Clone the Repository
```bash
git clone [https://github.com/Annyjerry/source-code-analyzer.git](https://github.com/your-username/source-code-analyzer.git)
cd source-code-analyzer

## Install Dependencies
```bash
pip install -r requirements.txt