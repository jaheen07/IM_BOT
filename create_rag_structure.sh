#!/bin/bash

# Change this to your FastAPI project root if needed
PROJECT_ROOT="."

mkdir -p $PROJECT_ROOT/{config,data/{raw/{bangla,english},processed/{bangla,english},split/{bangla,english}},preprocess,splitter,vector_store,finetune,guardrails,inference,notebooks,scripts,tests}

# Create essential Python files
touch $PROJECT_ROOT/config/{settings.yaml,constants.py}
touch $PROJECT_ROOT/preprocess/{__init__.py,cleaner.py,language_detector.py,pipeline.py}
touch $PROJECT_ROOT/splitter/{__init__.py,text_splitter.py,splitter_utils.py}
touch $PROJECT_ROOT/vector_store/{__init__.py,embedder.py,store.py,retriever.py}
touch $PROJECT_ROOT/finetune/{__init__.py,prepare_dataset.py,train_model.py,evaluation.py}
touch $PROJECT_ROOT/guardrails/{__init__.py,profanity_filter.py,hallucination_checker.py,guardrails_engine.py}
touch $PROJECT_ROOT/inference/{__init__.py,predictor.py,pipeline.py,api.py}
touch $PROJECT_ROOT/notebooks/{exploration.ipynb,debug.ipynb}
touch $PROJECT_ROOT/scripts/{preprocess_all.py,build_vectorstore.py,run_finetune.py,run_inference.py,test_guardrails.py}
touch $PROJECT_ROOT/tests/{test_preprocess.py,test_splitter.py,test_vector_store.py,test_predictor.py}
touch $PROJECT_ROOT/{requirements.txt,README.md,.env}

echo "✅ RAG pipeline folder structure created inside '$PROJECT_ROOT'"