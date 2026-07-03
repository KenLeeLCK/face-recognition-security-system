# Security-Oriented Face Recognition System

This project implements a security-oriented open-set face recognition system based on visual face recognition.

## Project Overview

The system is designed for security scenarios where preventing unknown individuals from being falsely accepted is the primary objective. The pipeline includes face detection, alignment, embedding extraction, database construction, and open-set recognition.

## Main Pipeline

1. Face detection using MTCNN
2. Face cropping and alignment
3. Face embedding extraction
4. Face database construction
5. Open-set identity recognition
6. Threshold-based stranger rejection

## Key Improvements

Compared with the original mid-term version, this project introduces:

- Quality-weighted multi-template embedding
- Improved template add / modify / delete workflow
- Structured `.npz` database design
- Evaluation with FAR, FRR, TAR, precision, F1-score, and ROC curve

## Security-Oriented Evaluation

In security scenarios, the main priority is to reduce false acceptance of unknown individuals. Therefore, threshold selection focuses on keeping FAR as low as possible while maintaining acceptable recognition recall.

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt