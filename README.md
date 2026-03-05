# 🧪 Carbon Dots LLM Extraction Pipeline

This repository contains the source code and documentation for an automated pipeline designed for data mining complex scientific literature. The project focuses on the structured extraction of synthesis parameters, morphology, and optical properties of Carbon Dots (CDs) using Large Language Models (LLMs).

## 🎯 Project Objective

The literature on nanomaterials is growing exponentially, making manual review unsustainable. This project solves this bottleneck through a pipeline that reads original scientific articles (in .xml format) and extracts precise data into a structured format (JSON), paving the way for the construction of Knowledge Graphs.

## ⚙️ Architecture and Methodology

The pipeline was architected into independent and specialized modules:

Pre-processing and Filtering (XML): Sanitization of the raw text from articles, removing noisy metadata (affiliations, bibliography) and converting complex tables into Markdown.

Auto-Prompting and Optimization: Use of LLMs (e.g., DeepSeek-R1) to generate and refine extraction instructions, aiming for maximum token efficiency and strict rule adherence.

Structured Extraction (JSON Schema): API requests utilizing response_format to force the extracting AI (e.g., GLM-5) to classify the data into 4 distinct blocks: General Info, Synthesis, Properties, and Applications.

Auditing with LLM-as-a-Judge: Rigorous evaluation of the generated extractions using the Evidently AI framework. High-context models (e.g., Grok-4.1-fast) cross-reference the extraction with the original article using Few-Shot Learning and a "Prime Directive" (Quote-Before-Criticize) to mitigate hallucinations.

## 📂 Repository Structure  

API-teste.py: Main script containing all the logic for XML processing, API communication (OpenRouter), extraction, and evaluation via Evidently.

main.tex & bib.bib: LaTeX source code of the project's complete technical report, including references in BibTeX format.

imgs/: Directory containing the generated graphs for metrics analysis (Benchmarking, Strict JSON, Prompt Length, etc.).

.gitignore: Filter for temporary LaTeX compiler files and Python cache.

## 🚀 Getting Started

Prerequisites

Python 3.11+

Libraries: requests, pandas, xml.etree, evidently, scikit-learn.

A valid API key from OpenRouter.

Steps

Clone the repository:

git clone [https://github.com/YOUR_USERNAME/carbon-dots-llm-extraction.git](https://github.com/YOUR_USERNAME/carbon-dots-llm-extraction.git)


Install the dependencies:

pip install pandas requests evidently scikit-learn


In the API-teste.py file, securely insert your OpenRouter API key (using environment variables is highly recommended).

Organize your test articles (.xml format) in the corresponding folder indicated in the script's configuration classes.

Run the main script:

python API-teste.py


📄 License and Authorship

Project developed by Amauri Jardim de Paula as part of research on the application of Generative AI in Materials Science and Nanotechnology.
