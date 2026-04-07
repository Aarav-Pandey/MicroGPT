# MicroGPT
Custom LLM Project Implementation

This repository contains the implementation of a **local Retrieval-Augmented Generation (RAG) system** with supporting tools and an agent controller. The project currently includes the following Python scripts:

## Project Files

- `rag_master.py`  
  Handles document retrieval and vector storage for the RAG workflow.

- `math_tool.py`  
  Provides mathematical and computational utilities for the agent.

- `agent_controller.py`  
  Controls the LLM agent workflow, managing reasoning, actions, and tool usage.

- `app.py`  
  Entry point for running the application. Integrates all modules to demonstrate the RAG-powered agent.

## Requirements

The project requires Python 3.10+ and the following dependencies:

```bash
pip install -r requirements.txt
