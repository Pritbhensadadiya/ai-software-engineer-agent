# AI Software Engineer Agent

An AI-powered software engineering system that can analyze a repository, retrieve relevant code, plan implementation work, modify code through tools, run tests, debug failures, review changes, and evaluate the overall workflow.

The project is designed as a **multi-agent software engineering workflow** rather than a single LLM prompt.

---

## 🚀 Overview

The AI Software Engineer Agent coordinates multiple specialized agents:

```text
User Task
    │
    ▼
┌─────────────┐
│   Planner   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Developer  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Tester   │
└──────┬──────┘
       │
       ├──────────────► Tests Pass
       │                     │
       │                     ▼
       │               ┌─────────────┐
       │               │   Reviewer  │
       │               └──────┬──────┘
       │                      │
       │                      ▼
       │                  Completed
       │
       ▼
┌─────────────┐
│   Debugger  │
└──────┬──────┘
       │
       └──────────────► Re-test

✨ Features
Multi-Agent Architecture
The system separates software engineering responsibilities across specialized agents:
- Planner Agent — analyzes the task and creates an implementation plan.
- Developer Agent — implements the plan using repository tools.
- Test Agent — creates/updates tests and executes them.
- Debugger Agent — investigates failures and attempts corrective changes.
- Code Reviewer Agent — reviews implementation changes and identifies issues.
- Orchestrator — coordinates the complete workflow.

🔎 Repository Understanding
The system can inspect an existing repository before making changes.
Capabilities include:
- Repository file discovery
- File reading
- Code searching
- Repository analysis
- Code chunking
- Code indexing
- Semantic code retrieval
- Incremental index management
This allows agents to work with the existing codebase instead of relying only on the user's task description.

🧠 Code Retrieval
The project contains a code indexing and retrieval pipeline:
Repository
    │
    ▼
Code Chunker
    │
    ▼
Code Indexer
    │
    ▼
Embeddings
    │
    ▼
Vector Store
    │
    ▼
Code Retriever
    │
    ▼
Relevant Repository Context

The generated repository index is managed separately from the source code.


🛠️ Tool System
Agents interact with the repository through registered tools.
Available capabilities include:
- list_files
- read_file
- write_file
- search_code
- apply_patch
- run_tests
- analyze_repository
- retrieve_code
The tool registry provides a controlled interface between the agents and the repository.


🔄 Debugging & Self-Correction
When tests fail, the workflow can enter a debugging loop:
Test Failure
     │
     ▼
  Debugger
     │
     ▼
Investigate Failure
     │
     ▼
Apply Correction
     │
     ▼
Run Tests Again
     │
     ├── Pass ──► Continue Workflow
     │
     └── Fail ──► Retry Debugging

The workflow limits debugging attempts to prevent uncontrolled loops.


👀 Code Review
After implementation and testing, the Code Reviewer analyzes the resulting implementation.
The reviewer can identify issues such as:
- overly generic exceptions
- potentially unintended return values
- suspicious implementation patterns
- other configured code-quality concerns
The review produces structured findings and an approval/change-request status.


📊 Workflow Evaluation
The project includes an evaluation layer that checks whether the workflow completed successfully.
Evaluation considers:
- Workflow completion
- Required agent stages
- Debugger invocation when necessary
- Final workflow output
Evaluation results include:
- Pass/fail status
- Score
- Individual checks
Multiple workflow scenarios are also evaluated, including successful workflows, workflows requiring debugging, and unrecoverable failures.


🏗️ Architecture
                         ┌───────────────────┐
                         │       User        │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │   Main / CLI      │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │     Workflow      │
                         └─────────┬─────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │         Orchestrator        │
                    └──────────────┬──────────────┘
                                   │
             ┌─────────────────────┼─────────────────────┐
             │                     │                     │
             ▼                     ▼                     ▼
        ┌─────────┐          ┌───────────┐        ┌─────────┐
        │ Planner │          │ Developer │        │ Tester  │
        └─────────┘          └───────────┘        └────┬────┘
                                                        │
                                                  ┌─────┴─────┐
                                                  │           │
                                               Pass        Failure
                                                  │           │
                                                  │           ▼
                                                  │      ┌──────────┐
                                                  │      │ Debugger │
                                                  │      └────┬─────┘
                                                  │           │
                                                  │        Re-test
                                                  │           │
                                                  └─────┬─────┘
                                                        │
                                                        ▼
                                                  ┌──────────┐
                                                  │ Reviewer │
                                                  └────┬─────┘
                                                       │
                                                       ▼
                                                   Completed

Repository intelligence is available to the agents through the tool and retrieval layers.


📁 Project Structure
AI Software Engineer Agent/
│
├── agent.py
├── agent_factory.py
├── main.py
├── workflow.py
├── orchestrator.py
├── workflow_logger.py
│
├── planner_agent.py
├── developer_agent.py
├── test_agent.py
├── debugger_agent.py
├── code_reviewer_agent.py
│
├── debugger.py
├── code_reviewer.py
├── evaluation.py
├── evaluation_scenarios.py
│
├── repository_analyzer.py
├── code_chunker.py
├── code_indexer.py
├── code_index_manager.py
├── code_retriever.py
├── search_codebase.py
├── index_manifest.py
├── embeddings.py
├── vector_store.py
│
├── tools.py
├── tool_registry.py
├── models.py
├── llm.py
│
├── tests/
│
├── requirements.txt
├── .env.example
└── README.md


⚙️ Requirements
- Python 3.13+
- Google Gemini API access
- Virtual environment recommended
- Dependencies listed in requirements.txt

🔧 Installation
Clone or copy the repository and enter the project directory.
Create a virtual environment:
python -m venv .venv

Activate it on Windows:
.venv\Scripts\Activate.ps1

Install dependencies:
pip install -r requirements.txt


🔑 Gemini API Configuration
Create a .env file in the project root:
GEMINI_API_KEY=your_gemini_api_key

Do not commit the .env file to source control.
A template is provided as:
.env.example


▶️ Running the Application
Start the production CLI with:
python main.py

The application will ask for a software engineering task:
AI Software Engineer Agent — Production Workflow
Enter a software engineering task.

Task:

Example:
Explain how the repository code indexing system works.

The application initializes the Gemini client, loads/builds the repository code index, and starts the multi-agent workflow.


🧪 Running Tests
Run the complete test suite:
pytest -q

Current project status:
229 passed

The test suite covers the major components of the system, including:
- Repository tools
- Tool registry
- Agents
- Debugging
- Code review
- Orchestration
- Workflow execution
- Evaluation
- Production CLI
- Error handling
- Failure and recovery scenarios


🔐 Safety & Reliability
The system includes several protections around agent-driven repository modification.
Examples include:
- Project-root path validation
- Protection against accessing files outside the project
- File-size limits
- Ignored repository directories
- Controlled tool registration
- Tool argument validation
- Maximum tool-call limits
- Maximum workflow-stage limits
- Limited debugging attempts
- Verification after modifications
- Structured workflow failure results
The Developer Agent is instructed to prefer targeted changes instead of blindly rewriting complete files.


🔄 Complete Workflow
A typical software engineering request follows this process:
1. User submits a task
Add a feature to the repository.

2. Planner analyzes the task
The Planner identifies:
- relevant files
- implementation steps
- dependencies
- testing requirements
3. Developer implements the plan
The Developer:
- retrieves relevant code
- inspects existing implementation
- applies targeted changes
- verifies modifications
4. Tester validates the implementation
Tests are created or updated and executed.
5. Debugger handles failures
If tests fail, the Debugger investigates and attempts a correction.
6. Reviewer examines the implementation
The Code Reviewer evaluates the resulting changes.
7. Evaluation measures the workflow
The evaluation layer determines whether the workflow completed successfully.


📈 Reliability Controls
The orchestration layer includes limits to prevent uncontrolled agent behavior.
Current controls include:
Maximum workflow stages: 10
Maximum debugger attempts: 2
Maximum agent tool calls: 20

These constraints help prevent infinite execution loops and uncontrolled tool usage.


🧪 Testing Philosophy
The project emphasizes testing the workflow behavior, not only individual functions.
Tests cover scenarios such as:
Successful workflow
       │
       ▼
Planner → Developer → Tester → Reviewer

and:
Planner → Developer → Tester
                         │
                       Failure
                         │
                         ▼
                      Debugger
                         │
                       Fix
                         │
                         ▼
                       Re-test
                         │
                         ▼
                      Reviewer

Persistent failures are also tested to ensure the system stops safely instead of continuing indefinitely.


🚧 Current Limitation
The application depends on Gemini API availability and quota.
If the configured Gemini API key has exhausted its available quota, the application detects the quota error and exits gracefully with a user-facing message.
This does not affect the automated test suite because the tests use controlled test doubles rather than requiring live Gemini API calls.


🚀 Future Improvements
Potential future improvements include:
- More sophisticated code-review rules
- Better automated test generation
- More advanced failure diagnosis
- Improved repository retrieval ranking
- Richer workflow observability
- Persistent workflow history
- More detailed evaluation metrics
- Parallel agent execution
- Human approval checkpoints
- Production web interface
- Additional LLM/provider support


🎯 Project Goals
This project demonstrates how an AI system can move beyond simple question answering toward an agentic software engineering workflow.
The central idea is:
Understand
   ↓
Plan
   ↓
Implement
   ↓
Test
   ↓
Debug
   ↓
Review
   ↓
Evaluate

Instead of treating an LLM as a standalone chatbot, the project combines:
- LLM reasoning
- Tool calling
- Repository analysis
- Code retrieval
- Multi-agent orchestration
- Automated testing
- Self-correction
- Code review
- Workflow evaluation
into a single software engineering system.


📊 Project Status
Status: Production-ready project baseline
Automated tests: 229 passed
Multi-agent workflow: Implemented
Repository indexing: Implemented
Code retrieval: Implemented
Tool system: Implemented
Testing agent: Implemented
Debugger: Implemented
Code reviewer: Implemented
Workflow evaluation: Implemented
Production CLI: Implemented

The live Gemini workflow additionally requires a Gemini API key with available quota.