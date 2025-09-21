# Challenge-Driven Swarm Intelligence Research Agent

## Overview

This research agent implements a **challenge-driven swarm intelligence system** that automatically explores complex research questions through hierarchical problem decomposition and collaborative agent processing. Unlike traditional search or Q&A systems, it actively identifies obstacles, systematically resolves them, and builds a comprehensive understanding through iterative refinement.

## Core Philosophy

### 1. **Challenge-Centric Approach**
- Research progress is fundamentally about **identifying and resolving challenges**
- Every question contains implicit obstacles that must be explicitly surfaced
- Solutions emerge from systematically addressing these challenges at increasing levels of specificity

### 2. **Swarm Intelligence**
- Multiple specialized agents work together, each with distinct cognitive roles
- No single agent has complete understanding; intelligence emerges from collaboration
- Agents challenge, validate, and refine each other's outputs

### 3. **Hierarchical Resolution**
- Complex problems decompose into simpler sub-challenges
- Resolution flows upward: solving children enables parent resolution
- The system builds a **challenge resolution tree** that maps the entire problem space

## Architecture

### Research Tree Structure
```
🎯 Root Challenge: "How to cure genetic diseases?"
  └→ ⚡ Challenge: "How to deliver CRISPR to neurons?"
      └→ ⚡ Sub: "Which AAV serotype crosses BBB?"
          └→ ✓ Finding: "AAV9 shows 40% efficiency"
      └→ ⚡ Sub: "How to prevent immune response?"
          └→ ◐ Partial: "Immunosuppressants help but..."
```

### Agent Roles

1. **Challenge Extractor**: Identifies problems, obstacles, and unknowns in text
2. **Challenge Specifier**: Breaks vague challenges into concrete, researchable sub-problems
3. **Resolution Agent**: Attempts to resolve challenges using available findings
4. **Analyzer**: Deep semantic analysis of responses
5. **Extractor**: Pulls concrete findings and facts
6. **Comparator**: Evaluates relevance to goals
7. **Challenger**: Generates critical questions to test assumptions
8. **Evaluator**: Assesses iteration quality and completeness
9. **Hypothesis**: Generates testable hypotheses from resolved challenges

### Processing Flow

1. **Strategic Direction**: Remote LLM provides high-level research strategy
2. **External Knowledge**: Fetches relevant papers from ArXiv
3. **Swarm Processing**: Local agents collaboratively process information
4. **Challenge Extraction**: Identifies all obstacles and problems
5. **Resolution Attempts**: Tries to resolve challenges with current knowledge
6. **Challenge Specification**: Breaks unresolved challenges into sub-problems
7. **Quality Evaluation**: Assesses whether sufficient progress has been made
8. **Tree Expansion**: Creates child nodes for unresolved challenges
9. **Recursive Resolution**: Process continues until challenges are resolved or depth limit reached

## Key Features

### Intelligent Challenge Management
- Automatically identifies technical, knowledge, resource, and methodological challenges
- Tracks resolution status (open, partial, resolved, blocked)
- Maintains challenge hierarchy and dependencies
- Identifies blocking challenges that prevent progress

### Multi-Model Architecture
- **Remote LLM** (Groq/OpenAI): Strategic planning and direction
- **Local LLM** (Llama.cpp/Transformers/Ollama): Rapid swarm agent processing
- **Embeddings** (optional): Semantic similarity and clustering
- **Neo4j Graph DB**: Persistent knowledge storage and relationship mapping

### Quality-Driven Iteration
- Each iteration evaluates completeness, accuracy, novelty, and progress
- Continues processing until quality threshold reached
- Stores only high-confidence findings to knowledge graph

## Installation & Setup

### Prerequisites
```bash
# Required
pip install neo4j arxiv

# Local LLM (choose one)
pip install llama-cpp-python  # Recommended: Downloads Mistral 7B automatically
# OR
pip install transformers torch  # Uses microsoft/phi-2
# OR
brew install ollama && ollama pull mistral  # macOS/Linux

# Optional
pip install sentence-transformers  # For embeddings
pip install python-dotenv  # For .env file support
```

### Environment Variables (.env)

```bash
# REQUIRED: Investigation context
THINKTICA_INVESTIGATION_ID=your_investigation_id
THINKTICA_RESEARCH_QUESTION="Your research question here"
THINKTICA_WORKSPACE=default

# Neo4j Database (optional but recommended)
THINKTICA_NEO4J_URI=bolt://localhost:8228
THINKTICA_NEO4J_USER=neo4j
THINKTICA_NEO4J_PASS=password

# Remote LLM for strategic planning (optional)
LLM_PROVIDER=groq  # or openai, anthropic
GROQ_API_KEY=your_groq_api_key
# OR
OPENAI_API_KEY=your_openai_key

# Model selection (optional)
LOCAL_MODEL_PATH=/path/to/model.gguf  # Custom GGUF model
EMBEDDING_MODEL=all-MiniLM-L6-v2  # Sentence transformer model
```

## Usage

### Basic Usage
```bash
# Set up investigation context
export THINKTICA_INVESTIGATION_ID="inv_12345"
export THINKTICA_RESEARCH_QUESTION="How can CRISPR cure genetic diseases?"

# Run the agent
python challenge_swarm_agent.py
```

### With Thinktica Framework
```bash
# Create investigation
thinktica workspace investigation new default "How can CRISPR cure genetic diseases?"

# Start with investigation context
thinktica start default --investigation inv_12345
```

### Programmatic Usage
```python
import asyncio
from challenge_swarm_agent import ChallengeTreeResearchAgent

async def research():
    agent = ChallengeTreeResearchAgent({
        'remote_provider': 'groq',
        'max_depth': 3,
        'neo4j_uri': 'bolt://localhost:8228'
    })
    
    await agent.research(
        goal="How can we achieve room-temperature superconductivity?",
        max_depth=3
    )

asyncio.run(research())
```

## Output

The system produces:

1. **Challenge Resolution Tree**: Hierarchical view of all challenges and their resolution status
2. **Top Discoveries**: High-confidence findings ranked by relevance
3. **Resolution Analytics**: Statistics on challenge resolution by depth
4. **Key Blockers**: Critical unresolved challenges preventing progress
5. **Neo4j Knowledge Graph**: Persistent storage of all findings, challenges, and relationships

## How It Works

### 1. Challenge Identification
The system scans text for:
- Explicit obstacles ("difficult to...", "challenge is...")
- Knowledge gaps ("unknown whether...", "unclear how...")
- Resource limitations ("requires...", "needs...")
- Methodological issues ("how to...")

### 2. Resolution Strategy
For each challenge:
1. Attempt resolution with current knowledge
2. If unresolvable, break into specific sub-challenges
3. Research sub-challenges independently
4. Combine sub-solutions to resolve parent

### 3. Quality Control
- Confidence scores on all findings
- Multiple agents validate each other
- Resolution requires >70% confidence
- Low-quality iterations trigger additional processing

### 4. Knowledge Building
- Findings accumulate across iterations
- Successful resolutions enable parent challenges
- Knowledge graph captures relationships
- System learns which approaches work

## Advanced Features

### Blocking Challenge Detection
Identifies challenges that prevent multiple other challenges from being resolved, helping prioritize research efforts.

### Resolution Paths
Tracks successful paths from root question to specific solutions, learning effective decomposition strategies.

### Hypothesis Generation
When challenges are resolved, generates testable hypotheses that connect findings and suggest new research directions.

### External Knowledge Integration
Automatically fetches and integrates relevant scientific papers from ArXiv to augment local knowledge.

## Limitations & Considerations

- **API Rate Limits**: Remote LLMs have rate limits; system includes automatic throttling
- **Local Model Size**: Llama.cpp models are ~4GB; ensure sufficient disk space
- **Processing Time**: Deep research can take 5-15 minutes depending on complexity
- **Memory Usage**: Local models require 4-8GB RAM
- **Neo4j Optional**: Works without database but won't persist findings

## Philosophy Notes

This system embodies the belief that **intelligence is fundamentally about problem-solving through challenge resolution**. Rather than trying to directly answer questions, it:

1. Maps the challenge landscape
2. Identifies critical obstacles
3. Systematically resolves them
4. Builds understanding through resolution

The swarm approach ensures **robustness through redundancy** - no single agent failure breaks the system. The hierarchical structure provides **natural progress tracking** - you can see exactly which challenges remain and why.

Most importantly, this creates **explainable AI reasoning** - every conclusion traces back through resolved challenges, making the logic transparent and verifiable.