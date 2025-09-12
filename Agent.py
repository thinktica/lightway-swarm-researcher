#!/usr/bin/env python3
"""
Challenge-Driven Swarm Intelligence Research Agent
===================================================
Enhanced with hierarchical challenge resolution system
"""

import asyncio
import os
import sys
import json
import hashlib
import logging
import subprocess
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import time
import re
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Loaded .env file")
except ImportError:
    logger.info("python-dotenv not installed, using system environment variables")

# Initialize availability flags
EMBEDDINGS_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
LLAMACPP_AVAILABLE = False
SentenceTransformer = None

# Core imports with auto-installation
try:
    from neo4j import GraphDatabase
except ImportError:
    print("Installing neo4j...")
    subprocess.run([sys.executable, "-m", "pip", "install", "neo4j"])
    from neo4j import GraphDatabase

try:
    import arxiv
except ImportError:
    print("Installing arxiv...")
    subprocess.run([sys.executable, "-m", "pip", "install", "arxiv"])
    import arxiv

# Try optional imports
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
    logger.info("Sentence-transformers loaded successfully")
except (ImportError, Exception) as e:
    logger.info("Sentence-transformers not available (optional)")
    EMBEDDINGS_AVAILABLE = False
    SentenceTransformer = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers available for backup LLM")
except ImportError:
    logger.info("Transformers not available")
    TRANSFORMERS_AVAILABLE = False

try:
    from llama_cpp import Llama
    LLAMACPP_AVAILABLE = True
    logger.info("llama-cpp available")
except ImportError:
    logger.info("llama-cpp not available")
    LLAMACPP_AVAILABLE = False


class NodeType(Enum):
    """Types of nodes in the research tree"""
    ROOT_QUESTION = "root_question"
    CHALLENGE = "challenge"
    SUB_CHALLENGE = "sub_challenge"
    HYPOTHESIS = "hypothesis"
    FINDING = "finding"
    SOLUTION = "solution"
    DEAD_END = "dead_end"


class NodeStatus(Enum):
    """Status of research nodes"""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    PARTIALLY_RESOLVED = "partially_resolved"
    BLOCKED = "blocked"
    ABANDONED = "abandoned"


class SwarmRole(Enum):
    """Roles for swarm agents"""
    ANALYZER = "analyzer"
    EXTRACTOR = "extractor"
    COMPARATOR = "comparator"
    EMBEDDER = "embedder"
    QUERIER = "querier"
    CHALLENGER = "challenger"
    EVALUATOR = "evaluator"
    HYPOTHESIS = "hypothesis"
    CHALLENGE_EXTRACTOR = "challenge_extractor"
    CHALLENGE_SPECIFIER = "challenge_specifier"
    RESOLUTION_AGENT = "resolution_agent"


@dataclass
class ResearchNode:
    """Enhanced node in the research tree"""
    id: str
    type: NodeType
    content: str
    status: NodeStatus
    parent_id: Optional[str]
    depth: int
    confidence: float = 0.0
    priority: float = 0.5
    findings: List[Dict] = field(default_factory=list)
    challenges: List[Dict] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    resolution_evidence: Optional[Dict] = None
    blockers: List[str] = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None


@dataclass
class Challenge:
    """Represents a challenge to be resolved"""
    id: str
    statement: str
    type: str  # technical/knowledge/resource/method
    priority: str  # high/medium/low
    parent_context: str
    sub_challenges: List[str] = field(default_factory=list)
    resolution_attempts: List[Dict] = field(default_factory=list)
    status: NodeStatus = NodeStatus.OPEN


@dataclass
class SwarmContext:
    """Context shared across swarm agents"""
    global_goal: str
    current_node: ResearchNode
    remote_response: str
    iteration: int = 0
    max_iterations: int = 100
    findings_buffer: List[Dict] = field(default_factory=list)
    challenges_buffer: List[Challenge] = field(default_factory=list)
    quality_scores: List[float] = field(default_factory=list)
    neo4j_cache: Dict = field(default_factory=dict)
    embeddings_cache: Dict = field(default_factory=dict)
    resolution_history: List[Dict] = field(default_factory=list)


class LocalLLMBackend:
    """Unified backend for local LLMs"""
    
    def __init__(self):
        self.backend = None
        self.model = None
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the best available backend"""
        
        if self._try_llamacpp():
            self.backend = 'llamacpp'
            logger.info("Using Llama.cpp backend")
            return
        
        if self._try_transformers():
            self.backend = 'transformers'
            logger.info("Using Transformers backend")
            return
        
        if self._try_ollama():
            self.backend = 'ollama'
            logger.info("Using Ollama backend")
            return
        
        logger.warning("No local LLM available - using mock responses")
        self.backend = 'mock'
    
    def _try_llamacpp(self) -> bool:
        """Try to use llama.cpp"""
        
        if not LLAMACPP_AVAILABLE:
            logger.info("Installing llama-cpp-python...")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "llama-cpp-python"],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    return False
            except Exception as e:
                logger.error(f"Could not install llama-cpp-python: {e}")
                return False
        
        try:
            from llama_cpp import Llama
        except ImportError:
            return False
        
        model_dir = Path.home() / ".cache" / "llm_models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        
        if not model_path.exists():
            logger.info("Downloading Mistral 7B Q4 model...")
            url = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
            
            try:
                import urllib.request
                
                def download_progress(block_num, block_size, total_size):
                    downloaded = block_num * block_size
                    percent = min(100, (downloaded / total_size) * 100)
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    bar_length = 40
                    filled = int(bar_length * percent / 100)
                    bar = '█' * filled + '-' * (bar_length - filled)
                    print(f'\rDownloading: |{bar}| {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)', end='', flush=True)
                
                urllib.request.urlretrieve(url, model_path, reporthook=download_progress)
                print()
                logger.info(f"Model downloaded to {model_path}")
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                return False
        
        try:
            logger.info("Loading Mistral 7B Q4 model...")
            self.model = Llama(
                model_path=str(model_path),
                n_ctx=2048,
                n_threads=min(8, os.cpu_count() or 4),
                n_gpu_layers=0,
                verbose=False,
                seed=42
            )
            
            test_response = self.model(
                "Hello, this is a test.",
                max_tokens=10,
                temperature=0.1
            )
            
            if test_response and 'choices' in test_response:
                logger.info("Model loaded successfully!")
                return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
        
        return False
    
    def _try_transformers(self) -> bool:
        """Try Transformers backend"""
        if not TRANSFORMERS_AVAILABLE:
            return False
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            
            model_name = "microsoft/phi-2"
            logger.info(f"Loading {model_name}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype="auto",
                low_cpu_mem_usage=True
            )
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7
            )
            
            return True
        except Exception as e:
            logger.debug(f"Could not load Transformers model: {e}")
            return False
    
    def _try_ollama(self) -> bool:
        """Try Ollama backend"""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                if "mistral" not in result.stdout:
                    logger.info("Pulling Mistral model for Ollama...")
                    subprocess.run(["ollama", "pull", "mistral"], capture_output=True)
                return True
        except:
            pass
        return False
    
    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        """Generate text using available backend"""
        
        if self.backend == 'llamacpp':
            return self._generate_llamacpp(prompt, max_tokens)
        elif self.backend == 'transformers':
            return self._generate_transformers(prompt, max_tokens)
        elif self.backend == 'ollama':
            return self._generate_ollama(prompt)
        else:
            return self._generate_mock(prompt)
    
    def _generate_llamacpp(self, prompt: str, max_tokens: int) -> str:
        """Generate using llama.cpp"""
        try:
            formatted_prompt = f"[INST] {prompt} [/INST]"
            
            response = self.model(
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                repeat_penalty=1.1,
                stop=["[INST]", "</s>", "\n\n\n"]
            )
            
            generated = response['choices'][0]['text'].strip()
            generated = generated.replace("[INST]", "").replace("[/INST]", "").strip()
            
            return generated
        except Exception as e:
            logger.error(f"llama.cpp generation error: {e}")
            return "[Generation failed]"
    
    def _generate_transformers(self, prompt: str, max_tokens: int) -> str:
        """Generate using Transformers"""
        try:
            outputs = self.pipeline(
                prompt,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated = outputs[0]['generated_text']
            if prompt in generated:
                generated = generated.replace(prompt, '').strip()
            
            return generated
        except Exception as e:
            logger.error(f"Transformers generation error: {e}")
            return "[Generation failed]"
    
    def _generate_ollama(self, prompt: str) -> str:
        """Generate using Ollama"""
        try:
            result = subprocess.run(
                ["ollama", "run", "mistral", prompt],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
        return "[Generation failed]"
    
    def _generate_mock(self, prompt: str) -> str:
        """Generate mock responses"""
        if "challenge" in prompt.lower() and "extract" in prompt.lower():
            return '[{"statement": "How to deliver CRISPR to neurons?", "type": "technical", "priority": "high", "parent_context": "Gene therapy"}]'
        elif "specify" in prompt.lower() and "challenge" in prompt.lower():
            return '[{"statement": "Which AAV serotype for neural delivery?", "specificity": "high"}, {"statement": "How to prevent immune response?", "specificity": "high"}]'
        elif "resolution" in prompt.lower():
            return '{"can_resolve": false, "confidence": 0.3, "missing_info": ["AAV efficacy data"], "partial_solution": "Consider AAV9"}'
        elif "findings" in prompt.lower():
            return '[{"statement": "AAV9 crosses blood-brain barrier", "confidence": 0.8, "evidence": "Study X"}]'
        elif "hypothesis" in prompt.lower():
            return '[{"statement": "Combined AAV9-CRISPR approach", "rationale": "Best delivery", "test": "In vivo trial"}]'
        else:
            return "Mock response for testing"


_llm_backend = None

def get_llm_backend() -> LocalLLMBackend:
    """Get or create LLM backend singleton"""
    global _llm_backend
    if _llm_backend is None:
        _llm_backend = LocalLLMBackend()
    return _llm_backend


class LocalSwarmAgent:
    """Base class for local swarm agents"""
    
    def __init__(self, role: SwarmRole):
        self.role = role
        self.processing_count = 0
        self.llm_backend = get_llm_backend()
    
    def process(self, prompt: str) -> str:
        """Process with local LLM"""
        self.processing_count += 1
        
        role_prompt = f"You are a {self.role.value} agent. {prompt}"
        response = self.llm_backend.generate(role_prompt)
        
        if response and not response.startswith('['):
            return response
        
        return f"[{self.role.value} processing failed]"


class ChallengeExtractorAgent(LocalSwarmAgent):
    """Extract challenges from text"""
    
    def __init__(self):
        super().__init__(SwarmRole.CHALLENGE_EXTRACTOR)
    
    def extract_challenges(self, response: str, context: SwarmContext) -> List[Challenge]:
        """Extract challenges from response"""
        prompt = f"""
        Extract ALL challenges, problems, and obstacles from this text:
        {response[:1000]}
        
        Current research: {context.current_node.content}
        
        Look for:
        - Technical obstacles ("difficult to...", "challenge is...")
        - Knowledge gaps ("unknown whether...", "unclear how...")
        - Resource limitations ("requires...", "needs...")
        - Methodological issues ("how to...")
        - Implicit problems (things mentioned as needing solution)
        
        For each challenge provide:
        - statement: The specific challenge
        - type: technical/knowledge/resource/method
        - priority: high/medium/low
        - parent_context: What led to this challenge
        
        Output as JSON array.
        """
        
        result = self.process(prompt)
        
        challenges = []
        try:
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            if json_match:
                challenge_data = json.loads(json_match.group())
                for item in challenge_data[:10]:
                    challenge = Challenge(
                        id=hashlib.md5(f"{item.get('statement', '')}{datetime.now()}".encode()).hexdigest()[:12],
                        statement=item.get('statement', ''),
                        type=item.get('type', 'unknown'),
                        priority=item.get('priority', 'medium'),
                        parent_context=item.get('parent_context', '')
                    )
                    challenges.append(challenge)
        except:
            pass
        
        return challenges


class ChallengeSpecifierAgent(LocalSwarmAgent):
    """Break challenges into specific sub-challenges"""
    
    def __init__(self):
        super().__init__(SwarmRole.CHALLENGE_SPECIFIER)
    
    def specify_challenge(self, challenge: Challenge, context: SwarmContext) -> List[Challenge]:
        """Break down challenge into specific sub-challenges"""
        prompt = f"""
        Break down this challenge into SPECIFIC, actionable sub-problems:
        
        Challenge: {challenge.statement}
        Type: {challenge.type}
        Context: {challenge.parent_context}
        
        Generate 3-5 concrete sub-challenges that:
        1. Are more specific than the parent
        2. Can be individually researched
        3. Together would solve the parent challenge
        4. Have clear success criteria
        
        Example:
        Parent: "How to deliver CRISPR to brain?"
        Sub-challenges:
        - "Which AAV serotype crosses BBB best?"
        - "What's the maximum payload size for AAV9?"
        - "How to prevent AAV immune response?"
        - "What's the optimal promoter for neurons?"
        
        Output as JSON array with statement and specificity level.
        """
        
        result = self.process(prompt)
        
        sub_challenges = []
        try:
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            if json_match:
                sub_data = json.loads(json_match.group())
                for item in sub_data[:5]:
                    sub_challenge = Challenge(
                        id=hashlib.md5(f"{item.get('statement', '')}{datetime.now()}".encode()).hexdigest()[:12],
                        statement=item.get('statement', ''),
                        type=challenge.type,
                        priority=challenge.priority,
                        parent_context=challenge.statement
                    )
                    sub_challenges.append(sub_challenge)
        except:
            pass
        
        return sub_challenges


class ResolutionAgent(LocalSwarmAgent):
    """Attempt to resolve challenges"""
    
    def __init__(self):
        super().__init__(SwarmRole.RESOLUTION_AGENT)
    
    def attempt_resolution(self, challenge: Challenge, findings: List[Dict], context: SwarmContext) -> Dict:
        """Try to resolve challenge with available information"""
        findings_text = "\n".join([f"- {f.get('statement', '')}" for f in findings[:10]])
        
        prompt = f"""
        Attempt to resolve this challenge using available findings:
        
        Challenge: {challenge.statement}
        Type: {challenge.type}
        
        Available findings:
        {findings_text}
        
        Previous resolution attempts: {len(challenge.resolution_attempts)}
        
        Determine:
        1. Can this be resolved with current knowledge? (yes/no)
        2. If yes, what's the solution?
        3. If no, what specific information is missing?
        4. Confidence in resolution (0-1)
        5. Partial solutions available?
        
        Output as JSON with keys: can_resolve, solution, confidence, missing_info, partial_solution
        """
        
        result = self.process(prompt)
        
        try:
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {
            "can_resolve": False,
            "solution": None,
            "confidence": 0.0,
            "missing_info": [],
            "partial_solution": None
        }


class AnalyzerAgent(LocalSwarmAgent):
    """Deep analysis of responses"""
    
    def __init__(self):
        super().__init__(SwarmRole.ANALYZER)
    
    def analyze_response(self, response: str, context: SwarmContext) -> Dict:
        """Analyze response in detail"""
        prompt = f"""
        Perform detailed analysis of this response:
        {response[:1000]}
        
        Current node: {context.current_node.content}
        Goal: {context.global_goal}
        
        Extract:
        1. Key assertions with confidence
        2. Implicit assumptions
        3. Logical connections
        4. Contradictions
        5. Areas needing clarification
        
        Output as JSON.
        """
        
        result = self.process(prompt)
        
        try:
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {
            "assertions": [],
            "assumptions": [],
            "connections": [],
            "contradictions": [],
            "clarifications": []
        }


class ExtractorAgent(LocalSwarmAgent):
    """Extract findings from text"""
    
    def __init__(self):
        super().__init__(SwarmRole.EXTRACTOR)
    
    def extract_findings(self, response: str, analysis: Dict) -> List[Dict]:
        """Extract concrete findings"""
        prompt = f"""
        Extract concrete findings from this text:
        {response[:1000]}
        
        Analysis hints: {json.dumps(analysis.get('assertions', [])[:3])}
        
        For each finding:
        - statement: The finding
        - confidence: 0.0 to 1.0
        - evidence: Supporting text
        - category: discovery/hypothesis/fact/theory
        
        Output as JSON array.
        """
        
        result = self.process(prompt)
        
        try:
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            if json_match:
                findings = json.loads(json_match.group())
                return findings[:10]
        except:
            pass
        
        return []


class ComparatorAgent(LocalSwarmAgent):
    """Compare findings to goals"""
    
    def __init__(self):
        super().__init__(SwarmRole.COMPARATOR)
    
    def compare_to_goals(self, findings: List[Dict], context: SwarmContext) -> Dict:
        """Compare findings to research goals"""
        findings_text = "\n".join([f"- {f.get('statement', '')}" for f in findings[:5]])
        
        prompt = f"""
        Compare findings to goals:
        
        Findings:
        {findings_text}
        
        Current goal: {context.current_node.content}
        Global goal: {context.global_goal}
        
        Evaluate:
        1. Current relevance (0-1)
        2. Global relevance (0-1)
        3. Novelty (0-1)
        4. New direction suggested?
        
        Output as JSON.
        """
        
        result = self.process(prompt)
        
        try:
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {
            "current_relevance": 0.5,
            "global_relevance": 0.5,
            "novelty": 0.5,
            "new_direction": None
        }


class ChallengerAgent(LocalSwarmAgent):
    """Generate challenges from findings"""
    
    def __init__(self):
        super().__init__(SwarmRole.CHALLENGER)
    
    def generate_challenges(self, findings: List[Dict], context: SwarmContext) -> List[str]:
        """Generate challenging questions"""
        findings_summary = "\n".join([f.get('statement', '')[:100] for f in findings[:3]])
        
        prompt = f"""
        Generate challenging questions that:
        1. Test validity of findings
        2. Identify edge cases
        3. Expose assumptions
        4. Suggest alternatives
        
        Findings:
        {findings_summary}
        
        Context: {context.current_node.content}
        
        Generate 5 critical questions.
        """
        
        result = self.process(prompt)
        
        questions = []
        for line in result.split('\n'):
            line = line.strip()
            if line and '?' in line and len(line) > 20:
                questions.append(line)
        
        return questions[:5]


class EvaluatorAgent(LocalSwarmAgent):
    """Evaluate iteration quality"""
    
    def __init__(self, remote_llm=None):
        super().__init__(SwarmRole.EVALUATOR)
        self.quality_threshold = 0.5
        self.remote_llm = remote_llm
    
    def evaluate_iteration(self, context: SwarmContext) -> Dict:
        """Evaluate quality of current iteration"""
        findings_summary = json.dumps([
            {"statement": f.get("statement", "")[:100], 
             "confidence": f.get("confidence", 0)}
            for f in context.findings_buffer[:5]
        ])
        
        challenges_summary = json.dumps([
            {"statement": c.statement[:100],
             "status": c.status.value}
            for c in context.challenges_buffer[:5]
        ])
        
        prompt = f"""
        Evaluate this research iteration:
        
        Findings: {findings_summary}
        Challenges: {challenges_summary}
        Iteration: {context.iteration}
        Current: {context.current_node.content}
        
        Rate:
        1. Completeness (0-1)
        2. Accuracy (0-1)
        3. Novelty (0-1)
        4. Progress (0-1)
        
        Output as JSON.
        """
        
        result = self.process(prompt)
        
        evaluation = {'completeness': 0.5, 'accuracy': 0.5, 'novelty': 0.5, 'progress': 0.5}
        try:
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                evaluation.update(parsed)
        except:
            pass
        
        evaluation['quality_score'] = np.mean([
            evaluation.get('completeness', 0.5),
            evaluation.get('accuracy', 0.5),
            evaluation.get('novelty', 0.5),
            evaluation.get('progress', 0.5)
        ])
        
        evaluation['decision'] = 'complete' if evaluation['quality_score'] >= self.quality_threshold else 'continue'
        evaluation['ready_for_storage'] = evaluation['quality_score'] >= 0.5
        evaluation['generate_hypothesis'] = evaluation['quality_score'] >= 0.5
        
        logger.info(f"Quality: {evaluation['quality_score']:.2f}, Decision: {evaluation['decision']}")
        
        return evaluation


class HypothesisAgent(LocalSwarmAgent):
    """Generate hypotheses"""
    
    def __init__(self):
        super().__init__(SwarmRole.HYPOTHESIS)
    
    def generate_hypotheses(self, context: SwarmContext) -> List[Dict]:
        """Generate new hypotheses"""
        findings_text = "\n".join([
            f"- {f.get('statement', '')}"
            for f in context.findings_buffer[:10]
        ])
        
        resolved_challenges = [
            c for c in context.challenges_buffer 
            if c.status == NodeStatus.RESOLVED
        ]
        
        challenges_text = "\n".join([
            f"- RESOLVED: {c.statement}"
            for c in resolved_challenges[:5]
        ])
        
        prompt = f"""
        Generate hypotheses based on findings and resolved challenges:
        
        Findings:
        {findings_text}
        
        Resolved Challenges:
        {challenges_text}
        
        Goal: {context.global_goal}
        Current: {context.current_node.content}
        
        Generate 3 testable hypotheses that:
        1. Connect findings
        2. Build on resolved challenges
        3. Make predictions
        4. Are rigorous
        
        For each:
        - statement: Hypothesis
        - rationale: Why
        - test: How to validate
        
        Output as JSON array.
        """
        
        result = self.process(prompt)
        
        try:
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            if json_match:
                hypotheses = json.loads(json_match.group())
                return hypotheses[:3]
        except:
            pass
        
        return []


class SwarmOrchestrator:
    """Enhanced orchestrator with challenge management"""
    
    def __init__(self, neo4j_driver=None, remote_llm=None):
        self.driver = neo4j_driver
        self.remote_llm = remote_llm
        
        # Initialize all agents
        self.analyzer = AnalyzerAgent()
        self.extractor = ExtractorAgent()
        self.comparator = ComparatorAgent()
        self.challenger = ChallengerAgent()
        self.evaluator = EvaluatorAgent(remote_llm)
        self.hypothesis = HypothesisAgent()
        
        # New challenge agents
        self.challenge_extractor = ChallengeExtractorAgent()
        self.challenge_specifier = ChallengeSpecifierAgent()
        self.resolution_agent = ResolutionAgent()
        
        # Research tree
        self.research_tree: Dict[str, ResearchNode] = {}
        self.challenges_tree: Dict[str, Challenge] = {}
        self.current_node_id: Optional[str] = None
        
        logger.info("Enhanced swarm orchestrator initialized")
    
    async def process_remote_response(self, 
                                     response: str, 
                                     context: SwarmContext) -> Dict:
        """Process response with challenge-driven approach"""
        
        logger.info(f"=== SWARM ITERATION {context.iteration} ===")
        
        iteration_results = {
            'iteration': context.iteration,
            'findings': [],
            'challenges': [],
            'resolutions': [],
            'quality': 0.0,
            'decision': 'continue'
        }
        
        # Phase 1: Analyze response
        logger.info("Phase 1: Analyzing response...")
        analysis = self.analyzer.analyze_response(response, context)
        
        # Phase 2: Extract findings
        logger.info("Phase 2: Extracting findings...")
        findings = self.extractor.extract_findings(response, analysis)
        context.findings_buffer.extend(findings)
        iteration_results['findings'] = findings
        
        # Phase 3: Extract challenges (NEW)
        logger.info("Phase 3: Extracting challenges...")
        new_challenges = self.challenge_extractor.extract_challenges(response, context)
        context.challenges_buffer.extend(new_challenges)
        iteration_results['challenges'] = new_challenges
        
        # Phase 4: Attempt to resolve existing challenges
        logger.info("Phase 4: Attempting challenge resolution...")
        resolutions = []
        for challenge in context.challenges_buffer:
            if challenge.status == NodeStatus.OPEN:
                resolution = self.resolution_agent.attempt_resolution(
                    challenge, context.findings_buffer, context
                )
                
                if resolution['confidence'] > 0.7:
                    challenge.status = NodeStatus.RESOLVED
                    challenge.resolution_attempts.append(resolution)
                    resolutions.append({
                        'challenge': challenge.statement,
                        'solution': resolution['solution'],
                        'confidence': resolution['confidence']
                    })
                    logger.info(f"   RESOLVED: {challenge.statement[:50]}...")
                elif resolution['confidence'] > 0.3:
                    challenge.status = NodeStatus.PARTIALLY_RESOLVED
                    challenge.resolution_attempts.append(resolution)
        
        iteration_results['resolutions'] = resolutions
        
        # Phase 5: Specify unresolved challenges
        logger.info("Phase 5: Specifying unresolved challenges...")
        for challenge in context.challenges_buffer:
            if challenge.status == NodeStatus.OPEN and len(challenge.sub_challenges) == 0:
                sub_challenges = self.challenge_specifier.specify_challenge(challenge, context)
                challenge.sub_challenges = [sc.id for sc in sub_challenges]
                context.challenges_buffer.extend(sub_challenges)
                logger.info(f"   Created {len(sub_challenges)} sub-challenges for: {challenge.statement[:30]}...")
        
        # Phase 6: Compare to goals
        logger.info("Phase 6: Comparing to goals...")
        comparison = self.comparator.compare_to_goals(findings, context)
        
        # Phase 7: Generate new challenges from findings
        logger.info("Phase 7: Generating challenge questions...")
        challenge_questions = self.challenger.generate_challenges(findings, context)
        
        # Convert questions to challenges
        for question in challenge_questions:
            new_challenge = Challenge(
                id=hashlib.md5(f"{question}{datetime.now()}".encode()).hexdigest()[:12],
                statement=question,
                type="derived",
                priority="medium",
                parent_context="Generated from findings"
            )
            context.challenges_buffer.append(new_challenge)
        
        # Phase 8: Evaluate iteration
        logger.info("Phase 8: Evaluating quality...")
        evaluation = self.evaluator.evaluate_iteration(context)
        context.quality_scores.append(evaluation['quality_score'])
        
        # Phase 9: Store to Neo4j if quality threshold met
        if evaluation['ready_for_storage'] and self.driver:
            logger.info("Phase 9: Storing to database...")
            self._store_iteration_data(context, evaluation)
        
        # Phase 10: Generate hypotheses if appropriate
        if evaluation.get('generate_hypothesis', False):
            logger.info("Phase 10: Generating hypotheses...")
            new_hypotheses = self.hypothesis.generate_hypotheses(context)
            iteration_results['hypotheses'] = new_hypotheses
        
        # Update iteration results
        iteration_results['quality'] = evaluation['quality_score']
        iteration_results['decision'] = evaluation['decision']
        iteration_results['comparison'] = comparison
        
        logger.info(f"Iteration complete. Quality: {evaluation['quality_score']:.2f}")
        
        return iteration_results
    
    def _store_iteration_data(self, context: SwarmContext, evaluation: Dict) -> int:
        """Store findings and challenges to Neo4j"""
        if not self.driver:
            return 0
        
        stored_count = 0
        
        try:
            with self.driver.session() as session:
                # Store ResearchBranch
                session.run("""
                    MERGE (b:ResearchBranch {id: $id})
                    SET b.question = $question,
                        b.status = 'processing',
                        b.quality_score = $quality,
                        b.iteration = $iteration,
                        b.node_type = $node_type,
                        b.updated = datetime()
                """, id=context.current_node.id,
                    question=context.current_node.content,
                    quality=evaluation['quality_score'],
                    iteration=context.iteration,
                    node_type=context.current_node.type.value)
                
                # Store challenges with hierarchy
                for challenge in context.challenges_buffer:
                    try:
                        session.run("""
                            MERGE (c:Challenge {id: $id})
                            SET c.statement = $statement,
                                c.type = $type,
                                c.priority = $priority,
                                c.status = $status,
                                c.parent_context = $parent_context,
                                c.timestamp = datetime()
                        """, id=challenge.id,
                            statement=challenge.statement,
                            type=challenge.type,
                            priority=challenge.priority,
                            status=challenge.status.value,
                            parent_context=challenge.parent_context)
                        
                        # Link to research branch
                        session.run("""
                            MATCH (c:Challenge {id: $cid})
                            MATCH (b:ResearchBranch {id: $bid})
                            MERGE (c)-[:BLOCKS]->(b)
                        """, cid=challenge.id, bid=context.current_node.id)
                        
                        # Link sub-challenges
                        for sub_id in challenge.sub_challenges:
                            session.run("""
                                MATCH (parent:Challenge {id: $pid})
                                MATCH (child:Challenge {id: $cid})
                                MERGE (child)-[:REFINES]->(parent)
                            """, pid=challenge.id, cid=sub_id)
                        
                        # Store resolution if resolved
                        if challenge.status == NodeStatus.RESOLVED and challenge.resolution_attempts:
                            resolution = challenge.resolution_attempts[-1]
                            res_id = hashlib.md5(
                                f"{challenge.id}_resolution_{datetime.now()}".encode()
                            ).hexdigest()[:12]
                            
                            session.run("""
                                CREATE (r:Resolution {
                                    id: $id,
                                    solution: $solution,
                                    confidence: $confidence,
                                    timestamp: datetime()
                                })
                            """, id=res_id,
                                solution=resolution.get('solution', ''),
                                confidence=resolution.get('confidence', 0))
                            
                            session.run("""
                                MATCH (r:Resolution {id: $rid})
                                MATCH (c:Challenge {id: $cid})
                                CREATE (r)-[:RESOLVES]->(c)
                            """, rid=res_id, cid=challenge.id)
                        
                        stored_count += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to store challenge: {e}")
                
                # Store findings
                for finding in context.findings_buffer:
                    if finding.get('confidence', 0) >= 0.3:
                        try:
                            discovery_id = hashlib.md5(
                                f"{finding.get('statement', '')}{datetime.now()}".encode()
                            ).hexdigest()[:12]
                            
                            session.run("""
                                CREATE (d:Discovery {
                                    id: $id,
                                    statement: $statement,
                                    confidence_score: $confidence,
                                    category: $category,
                                    timestamp: datetime()
                                })
                            """, id=discovery_id,
                                statement=finding.get('statement', ''),
                                confidence=finding.get('confidence', 0.5),
                                category=finding.get('category', 'unknown'))
                            
                            session.run("""
                                MATCH (d:Discovery {id: $did})
                                MATCH (b:ResearchBranch {id: $bid})
                                CREATE (d)-[:PART_OF]->(b)
                            """, did=discovery_id, bid=context.current_node.id)
                            
                            stored_count += 1
                            
                        except Exception as e:
                            logger.error(f"Failed to store finding: {e}")
                
                logger.info(f"Stored {stored_count} items to Neo4j")
                
        except Exception as e:
            logger.error(f"Database error: {e}")
        
        return stored_count
    
    async def run_swarm_loop(self, 
                           response: str, 
                           node: ResearchNode,
                           global_goal: str,
                           max_iterations: int = 10) -> Dict:
        """Run swarm processing loop"""
        
        context = SwarmContext(
            global_goal=global_goal,
            current_node=node,
            remote_response=response,
            max_iterations=max_iterations
        )
        
        all_results = []
        
        for i in range(max_iterations):
            context.iteration = i + 1
            
            iteration_result = await self.process_remote_response(response, context)
            all_results.append(iteration_result)
            
            if iteration_result['decision'] == 'complete':
                logger.info(f"Quality threshold reached at iteration {i+1}")
                break
            
            # Refine response based on unresolved challenges
            unresolved = [c for c in context.challenges_buffer if c.status == NodeStatus.OPEN]
            if unresolved:
                challenge_text = "\n".join([f"- {c.statement}" for c in unresolved[:5]])
                response = response + f"\n\nUnresolved challenges:\n{challenge_text}"
        
        # Calculate challenge resolution rate
        total_challenges = len(context.challenges_buffer)
        resolved_challenges = len([c for c in context.challenges_buffer if c.status == NodeStatus.RESOLVED])
        resolution_rate = resolved_challenges / total_challenges if total_challenges > 0 else 0
        
        return {
            'node_id': node.id,
            'iterations_run': len(all_results),
            'final_quality': np.mean(context.quality_scores) if context.quality_scores else 0.0,
            'total_findings': len(context.findings_buffer),
            'total_challenges': total_challenges,
            'resolved_challenges': resolved_challenges,
            'resolution_rate': resolution_rate,
            'findings': context.findings_buffer,
            'challenges': [{'statement': c.statement, 'status': c.status.value} for c in context.challenges_buffer],
            'hypotheses': [r.get('hypotheses', []) for r in all_results if 'hypotheses' in r]
        }


class RemoteLLMOrchestrator:
    """Remote LLM for strategic decisions"""
    
    def __init__(self, provider: str = "groq", api_key: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        
        if not self.api_key:
            logger.warning(f"No API key for {provider}. Set {provider.upper()}_API_KEY")
    
    def get_strategic_direction(self, goal: str, current_state: Dict) -> str:
        """Get strategic direction from remote LLM"""
        
        # Include challenge information in prompt
        challenges_text = ""
        if 'challenges' in current_state:
            open_challenges = [c for c in current_state['challenges'] if c.get('status') == 'open']
            if open_challenges:
                challenges_text = "\n- Open challenges: " + ", ".join([c['statement'][:50] for c in open_challenges[:3]])
        
        prompt = f"""
        You are a genius research strategist solving complex problems.
        
        RESEARCH GOAL: {goal}
        
        CURRENT STATE:
        - Explored nodes: {current_state.get('explored_count', 0)}
        - Current depth: {current_state.get('current_depth', 0)}
        - Resolution rate: {current_state.get('resolution_rate', 0):.1%}
        {challenges_text}
        
        Provide:
        1. Assessment of progress
        2. Critical research directions
        3. Specific approaches to overcome challenges
        4. Key hypotheses to test
        5. Potential breakthrough paths
        
        Focus on concrete, actionable insights.
        """
        
        if self.provider == "groq":
            return self._call_groq(prompt)
        else:
            return "Strategic direction: Systematically resolve challenges."
    
    def _call_groq(self, prompt: str) -> str:
        """Call Groq API"""
        try:
            import requests
            import time
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 2000,
                "top_p": 0.95
            }
            
            max_retries = 3
            for attempt in range(max_retries):
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content']
                
                elif response.status_code == 429:
                    retry_after = response.headers.get('retry-after', '5')
                    wait_time = int(retry_after) + 1
                    logger.warning(f"Rate limit hit. Waiting {wait_time} seconds...")
                    
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                        continue
                    else:
                        return "[Rate limit exceeded]"
                else:
                    logger.error(f"Groq API error: {response.status_code}")
                    return "[Remote LLM error]"
            
        except Exception as e:
            logger.error(f"Groq error: {e}")
            return "[Remote LLM error]"


class ChallengeTreeResearchAgent:
    """Main agent with challenge-driven research"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.api_call_count = 0
        self.last_api_call = None
        
        self.remote_llm = RemoteLLMOrchestrator(
            provider=self.config.get('remote_provider', 'groq')
        )
        
        self.driver = self._connect_neo4j()
        self.swarm = SwarmOrchestrator(self.driver, self.remote_llm)
        
        self.tree_root: Optional[ResearchNode] = None
        self.challenge_queue = deque()
        self.node_queue = deque()
        
        self.arxiv_client = arxiv.Client()
        
        logger.info("Challenge-driven research agent initialized")
    
    def _connect_neo4j(self) -> Optional[GraphDatabase.driver]:
        """Connect to Neo4j"""
        try:
            driver = GraphDatabase.driver(
                "bolt://localhost:8228",
                auth=("neo4j", "password")
            )
            with driver.session() as session:
                session.run("RETURN 1")
            logger.info("Connected to Neo4j")
            return driver
        except:
            logger.warning("Neo4j not available")
            return None
    
    async def fetch_external_knowledge(self, query: str) -> List[Dict]:
        """Fetch from ArXiv"""
        papers = []
        try:
            search = arxiv.Search(
                query=query,
                max_results=3,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            for paper in self.arxiv_client.results(search):
                papers.append({
                    'title': paper.title,
                    'summary': paper.summary[:300],
                    'url': paper.entry_id
                })
        except Exception as e:
            logger.error(f"ArXiv error: {e}")
        
        return papers
    
    def create_research_node(self, 
                            content: str,
                            node_type: NodeType,
                            parent: Optional[ResearchNode] = None) -> ResearchNode:
        """Create a new research node"""
        node_id = hashlib.md5(f"{content}{datetime.now()}".encode()).hexdigest()[:12]
        
        node = ResearchNode(
            id=node_id,
            type=node_type,
            content=content,
            status=NodeStatus.OPEN,
            parent_id=parent.id if parent else None,
            depth=parent.depth + 1 if parent else 0
        )
        
        self.swarm.research_tree[node_id] = node
        
        if parent:
            parent.children.append(node_id)
        
        return node
    
    def check_parent_resolution(self, node: ResearchNode):
        """Check if parent can be resolved after child resolution"""
        if not node.parent_id:
            return
        
        parent = self.swarm.research_tree.get(node.parent_id)
        if not parent:
            return
        
        # Check if all children are resolved
        all_resolved = True
        for child_id in parent.children:
            child = self.swarm.research_tree.get(child_id)
            if child and child.status != NodeStatus.RESOLVED:
                all_resolved = False
                break
        
        if all_resolved:
            parent.status = NodeStatus.RESOLVED
            parent.resolved_at = datetime.now()
            logger.info(f"Parent challenge resolved: {parent.content[:50]}...")
            
            # Recursively check grandparent
            self.check_parent_resolution(parent)
    
    async def research(self, goal: str, max_depth: int = 3):
        """Main research loop with challenge focus"""
        
        print(f"\n{'='*80}")
        print(f"CHALLENGE-DRIVEN RESEARCH SYSTEM")
        print(f"Goal: {goal}")
        print(f"{'='*80}\n")
        
        # Create root challenge
        self.tree_root = self.create_research_node(
            content=f"Challenge: {goal}",
            node_type=NodeType.ROOT_QUESTION
        )
        self.node_queue.append(self.tree_root)
        
        nodes_explored = 0
        challenges_resolved = 0
        
        while self.node_queue and nodes_explored < 15:
            current_node = self.node_queue.popleft()
            
            if current_node.depth > max_depth:
                continue
            
            nodes_explored += 1
            
            print(f"\n{'='*60}")
            print(f"NODE {nodes_explored}: {current_node.content[:80]}...")
            print(f"Type: {current_node.type.value}, Depth: {current_node.depth}")
            print(f"Status: {current_node.status.value}")
            print(f"{'='*60}")
            
            # Get strategic direction
            print("\n1. REQUESTING STRATEGIC DIRECTION...")
            
            current_state = {
                'explored_count': nodes_explored,
                'current_depth': current_node.depth,
                'resolution_rate': challenges_resolved / max(nodes_explored, 1),
                'challenges': [{'statement': c.statement, 'status': c.status.value} 
                             for c in self.swarm.challenges_tree.values()][:5]
            }
            
            # Rate limiting
            if self.last_api_call:
                elapsed = time.time() - self.last_api_call
                if elapsed < 2.4:
                    wait_time = 2.4 - elapsed
                    await asyncio.sleep(wait_time)
            
            self.last_api_call = time.time()
            
            strategic_response = self.remote_llm.get_strategic_direction(
                current_node.content, current_state
            )
            
            print(f"   Direction received ({len(strategic_response)} chars)")
            
            # Fetch external knowledge
            print("\n2. FETCHING EXTERNAL KNOWLEDGE...")
            papers = await self.fetch_external_knowledge(current_node.content)
            print(f"   Found {len(papers)} papers")
            
            # Process through swarm
            print("\n3. CHALLENGE-AWARE SWARM PROCESSING...")
            
            swarm_results = await self.swarm.run_swarm_loop(
                response=strategic_response + "\n\nPapers:\n" + json.dumps(papers),
                node=current_node,
                global_goal=goal,
                max_iterations=5
            )
            
            print(f"   Iterations: {swarm_results['iterations_run']}")
            print(f"   Quality: {swarm_results['final_quality']:.2%}")
            print(f"   Findings: {swarm_results['total_findings']}")
            print(f"   Challenges: {swarm_results['total_challenges']}")
            print(f"   Resolution rate: {swarm_results['resolution_rate']:.1%}")
            
            # Update node
            current_node.findings = swarm_results['findings']
            current_node.challenges = swarm_results['challenges']
            current_node.confidence = swarm_results['final_quality']
            
            # Check if current challenge is resolved
            if swarm_results['resolution_rate'] > 0.7:
                current_node.status = NodeStatus.RESOLVED
                current_node.resolved_at = datetime.now()
                challenges_resolved += swarm_results['resolved_challenges']
                print(f"\n   ✓ RESOLVED: {current_node.content[:50]}...")
                
                # Check parent resolution
                self.check_parent_resolution(current_node)
            else:
                current_node.status = NodeStatus.PARTIALLY_RESOLVED
            
            # Create child nodes from unresolved challenges
            unresolved = [c for c in swarm_results['challenges'] if c['status'] == 'open']
            if unresolved:
                print("\n4. CREATING CHILD NODES FOR UNRESOLVED CHALLENGES...")
                for challenge in unresolved[:3]:
                    child = self.create_research_node(
                        content=challenge['statement'],
                        node_type=NodeType.SUB_CHALLENGE,
                        parent=current_node
                    )
                    self.node_queue.append(child)
                    print(f"   Added: {challenge['statement'][:60]}...")
            
            # Create nodes from hypotheses
            if swarm_results.get('hypotheses'):
                print("\n5. CREATING HYPOTHESIS NODES...")
                for hyp_batch in swarm_results['hypotheses']:
                    for hyp in hyp_batch[:2]:
                        if isinstance(hyp, dict) and 'statement' in hyp:
                            child = self.create_research_node(
                                content=hyp['statement'],
                                node_type=NodeType.HYPOTHESIS,
                                parent=current_node
                            )
                            self.node_queue.append(child)
                            print(f"   Added hypothesis: {hyp['statement'][:60]}...")
        
        # Final summary
        print(f"\n{'='*80}")
        print("RESEARCH COMPLETE - CHALLENGE RESOLUTION SUMMARY")
        print(f"{'='*80}")
        
        # Calculate final statistics
        total_nodes = len(self.swarm.research_tree)
        resolved_nodes = len([n for n in self.swarm.research_tree.values() 
                            if n.status == NodeStatus.RESOLVED])
        
        all_challenges = []
        for node in self.swarm.research_tree.values():
            all_challenges.extend(node.challenges)
        
        total_challenges = len(all_challenges)
        resolved_challenges = len([c for c in all_challenges if c['status'] == 'resolved'])
        
        print(f"\nNodes explored: {nodes_explored}")
        print(f"Nodes resolved: {resolved_nodes}/{total_nodes} ({resolved_nodes/max(total_nodes,1)*100:.1f}%)")
        print(f"Challenges identified: {total_challenges}")
        print(f"Challenges resolved: {resolved_challenges} ({resolved_challenges/max(total_challenges,1)*100:.1f}%)")
        print(f"Max depth reached: {max(n.depth for n in self.swarm.research_tree.values())}")
        
        # Print challenge hierarchy
        print("\n" + "="*60)
        print("CHALLENGE HIERARCHY")
        print("="*60)
        self.print_challenge_tree()
        
        # Store final state to Neo4j
        if self.driver:
            self.store_challenge_hierarchy()
        
        # Top findings
        all_findings = []
        for node in self.swarm.research_tree.values():
            all_findings.extend(node.findings)
        
        all_findings.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        print("\n" + "="*60)
        print("TOP DISCOVERIES")
        print("="*60)
        for finding in all_findings[:5]:
            print(f"[{finding.get('confidence', 0):.1%}] {finding.get('statement', '')[:100]}...")
    
    def print_challenge_tree(self, node_id: str = None, indent: int = 0):
        """Print the challenge hierarchy tree"""
        if node_id is None:
            # Start from root
            if not self.tree_root:
                return
            node_id = self.tree_root.id
        
        node = self.swarm.research_tree.get(node_id)
        if not node:
            return
        
        # Status symbols
        status_symbol = {
            NodeStatus.RESOLVED: "✓",
            NodeStatus.PARTIALLY_RESOLVED: "◐",
            NodeStatus.OPEN: "○",
            NodeStatus.BLOCKED: "✗",
            NodeStatus.ABANDONED: "⨯"
        }.get(node.status, "?")
        
        # Type symbols
        type_symbol = {
            NodeType.ROOT_QUESTION: "🎯",
            NodeType.CHALLENGE: "⚡",
            NodeType.SUB_CHALLENGE: "└→",
            NodeType.HYPOTHESIS: "💡",
            NodeType.SOLUTION: "🔧",
            NodeType.FINDING: "📊"
        }.get(node.type, "")
        
        # Print node
        indent_str = "  " * indent
        print(f"{indent_str}{status_symbol} {type_symbol} {node.content[:80]}...")
        
        # Print confidence if resolved
        if node.status == NodeStatus.RESOLVED:
            print(f"{indent_str}    ↳ Confidence: {node.confidence:.1%}")
        
        # Recursively print children
        for child_id in node.children:
            self.print_challenge_tree(child_id, indent + 1)
    
    def store_challenge_hierarchy(self):
        """Store the complete challenge hierarchy to Neo4j"""
        if not self.driver:
            return
        
        try:
            with self.driver.session() as session:
                # Create indexes for efficient querying
                session.run("""
                    CREATE INDEX challenge_hierarchy IF NOT EXISTS
                    FOR (n:ResearchNode) ON (n.id, n.depth, n.type)
                """)
                
                # Store all nodes with their relationships
                for node_id, node in self.swarm.research_tree.items():
                    # Store node
                    session.run("""
                        MERGE (n:ResearchNode {id: $id})
                        SET n.content = $content,
                            n.type = $type,
                            n.status = $status,
                            n.depth = $depth,
                            n.confidence = $confidence,
                            n.priority = $priority,
                            n.created_at = $created_at,
                            n.resolved_at = $resolved_at,
                            n.finding_count = $finding_count,
                            n.challenge_count = $challenge_count
                    """, 
                        id=node.id,
                        content=node.content,
                        type=node.type.value,
                        status=node.status.value,
                        depth=node.depth,
                        confidence=node.confidence,
                        priority=node.priority,
                        created_at=node.created_at,
                        resolved_at=node.resolved_at,
                        finding_count=len(node.findings),
                        challenge_count=len(node.challenges)
                    )
                    
                    # Store parent-child relationships
                    if node.parent_id:
                        session.run("""
                            MATCH (child:ResearchNode {id: $child_id})
                            MATCH (parent:ResearchNode {id: $parent_id})
                            MERGE (child)-[:REFINES]->(parent)
                            MERGE (parent)-[:HAS_CHILD]->(child)
                        """, child_id=node.id, parent_id=node.parent_id)
                    
                    # Store blockers
                    for blocker_id in node.blockers:
                        session.run("""
                            MATCH (n:ResearchNode {id: $node_id})
                            MATCH (b:ResearchNode {id: $blocker_id})
                            MERGE (n)-[:BLOCKED_BY]->(b)
                        """, node_id=node.id, blocker_id=blocker_id)
                    
                    # Store resolution evidence
                    if node.resolution_evidence:
                        res_id = hashlib.md5(
                            f"{node.id}_resolution_{datetime.now()}".encode()
                        ).hexdigest()[:12]
                        
                        session.run("""
                            CREATE (r:Resolution {
                                id: $id,
                                node_id: $node_id,
                                solution: $solution,
                                confidence: $confidence,
                                evidence: $evidence,
                                timestamp: datetime()
                            })
                        """, 
                            id=res_id,
                            node_id=node.id,
                            solution=node.resolution_evidence.get('solution', ''),
                            confidence=node.resolution_evidence.get('confidence', 0),
                            evidence=json.dumps(node.resolution_evidence.get('evidence', []))
                        )
                        
                        session.run("""
                            MATCH (r:Resolution {id: $res_id})
                            MATCH (n:ResearchNode {id: $node_id})
                            CREATE (r)-[:RESOLVES]->(n)
                        """, res_id=res_id, node_id=node.id)
                
                # Create aggregation relationships for UI queries
                session.run("""
                    MATCH (n:ResearchNode)
                    WHERE n.type = 'root_question'
                    SET n:RootChallenge
                """)
                
                session.run("""
                    MATCH (n:ResearchNode)
                    WHERE n.status = 'resolved'
                    SET n:Resolved
                """)
                
                session.run("""
                    MATCH (n:ResearchNode)
                    WHERE n.status IN ['open', 'blocked']
                    SET n:Unresolved
                """)
                
                logger.info(f"Stored {len(self.swarm.research_tree)} nodes to Neo4j with hierarchy")
                
        except Exception as e:
            logger.error(f"Failed to store hierarchy: {e}")
    
    def get_challenge_analytics(self) -> Dict:
        """Get analytics about challenge resolution for UI"""
        if not self.driver:
            return {}
        
        try:
            with self.driver.session() as session:
                # Get challenge resolution by depth
                result = session.run("""
                    MATCH (n:ResearchNode)
                    RETURN n.depth as depth,
                           n.status as status,
                           count(n) as count
                    ORDER BY depth, status
                """)
                
                depth_stats = defaultdict(lambda: {'resolved': 0, 'open': 0, 'total': 0})
                for record in result:
                    depth = record['depth']
                    status = record['status']
                    count = record['count']
                    depth_stats[depth]['total'] += count
                    if status == 'resolved':
                        depth_stats[depth]['resolved'] += count
                    elif status in ['open', 'blocked']:
                        depth_stats[depth]['open'] += count
                
                # Get resolution paths
                result = session.run("""
                    MATCH path = (child:ResearchNode)-[:REFINES*]->(root:RootChallenge)
                    WHERE child.status = 'resolved'
                    RETURN path,
                           length(path) as path_length,
                           child.confidence as confidence
                    ORDER BY confidence DESC
                    LIMIT 10
                """)
                
                resolution_paths = []
                for record in result:
                    path = record['path']
                    nodes = [node['content'][:50] for node in path.nodes]
                    resolution_paths.append({
                        'path': nodes,
                        'length': record['path_length'],
                        'confidence': record['confidence']
                    })
                
                # Get blocking challenges
                result = session.run("""
                    MATCH (blocked:ResearchNode)-[:BLOCKED_BY]->(blocker:ResearchNode)
                    WHERE blocker.status != 'resolved'
                    RETURN blocker.content as challenge,
                           count(blocked) as blocked_count,
                           blocker.depth as depth
                    ORDER BY blocked_count DESC
                    LIMIT 5
                """)
                
                key_blockers = []
                for record in result:
                    key_blockers.append({
                        'challenge': record['challenge'][:100],
                        'blocked_count': record['blocked_count'],
                        'depth': record['depth']
                    })
                
                return {
                    'depth_statistics': dict(depth_stats),
                    'resolution_paths': resolution_paths,
                    'key_blockers': key_blockers
                }
                
        except Exception as e:
            logger.error(f"Failed to get analytics: {e}")
            return {}


async def main():
    """Main entry point"""
    
    print("\n" + "="*80)
    print("CHALLENGE-DRIVEN SWARM INTELLIGENCE RESEARCH SYSTEM")
    print("="*80)
    
    # Check available backends
    backend = get_llm_backend()
    print(f"\nLocal LLM Backend: {backend.backend}")
    
    if backend.backend == 'mock':
        print("\nWARNING: No local LLM available. System will use mock responses.")
        print("\nTo enable local LLM, choose one option:")
        print("1. Let the script auto-install llama-cpp-python (recommended)")
        print("2. pip install transformers torch")
        print("3. Install Ollama from ollama.ai")
        
        if not input("\nContinue? (y/n): ").lower().startswith('y'):
            return
    
    # Configuration
    config = {
        'remote_provider': os.getenv('LLM_PROVIDER', 'groq'),
        'max_depth': 3,
        'max_iterations': 10
    }
    
    # Check remote LLM
    api_key_env = f"{config['remote_provider'].upper()}_API_KEY"
    if not os.getenv(api_key_env):
        print(f"\nWARNING: No {api_key_env} found.")
        print(f"Set it with: export {api_key_env}='your-api-key'")
        print("\nAvailable free providers:")
        print("  - groq: https://console.groq.com/keys")
        print("  - together: https://api.together.xyz/")
        print("\nSystem will use simplified strategic directions.")
    
    # Research question
    default_goal = "How can we develop safe and effective CRISPR-based treatments for neurodegenerative diseases?"
    
    print(f"\nDefault research goal: {default_goal}")
    custom = input("Enter custom goal (or press Enter for default): ").strip()
    goal = custom if custom else default_goal
    
    # Create and run agent
    agent = ChallengeTreeResearchAgent(config)
    await agent.research(goal, max_depth=3)
    
    # Show analytics
    print("\n" + "="*80)
    print("CHALLENGE RESOLUTION ANALYTICS")
    print("="*80)
    
    analytics = agent.get_challenge_analytics()
    
    if analytics.get('depth_statistics'):
        print("\nResolution by Depth:")
        for depth, stats in sorted(analytics['depth_statistics'].items()):
            resolved = stats['resolved']
            total = stats['total']
            rate = resolved / total * 100 if total > 0 else 0
            print(f"  Depth {depth}: {resolved}/{total} resolved ({rate:.1f}%)")
    
    if analytics.get('key_blockers'):
        print("\nKey Blocking Challenges:")
        for blocker in analytics['key_blockers']:
            print(f"  - {blocker['challenge']}")
            print(f"    Blocking {blocker['blocked_count']} other challenges (depth {blocker['depth']})")
    
    # Cleanup
    if agent.driver:
        agent.driver.close()


if __name__ == '__main__':
    # Print installation status
    print("\n" + "="*60)
    print("SYSTEM CHECK")
    print("="*60)
    print(f"✓ Core: Python {sys.version.split()[0]}")
    
    # Check components
    components = [
        ('Transformers', TRANSFORMERS_AVAILABLE, 'pip install transformers torch'),
        ('Llama.cpp', LLAMACPP_AVAILABLE, 'pip install llama-cpp-python'),
        ('Embeddings', EMBEDDINGS_AVAILABLE, 'pip install sentence-transformers')
    ]
    
    for name, available, install_cmd in components:
        status = '✓' if available else '✗'
        msg = 'Available' if available else install_cmd
        print(f"{status} {name}: {msg}")
    
    # Check Ollama
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        ollama_available = result.returncode == 0
    except:
        ollama_available = False
    print(f"{'✓' if ollama_available else '✗'} Ollama: {'Available' if ollama_available else 'Install from ollama.ai'}")
    
    # Check Neo4j
    try:
        test_driver = GraphDatabase.driver("bolt://localhost:8228", auth=("neo4j", "password"))
        with test_driver.session() as session:
            session.run("RETURN 1")
        neo4j_available = True
        test_driver.close()
    except:
        neo4j_available = False
    print(f"{'✓' if neo4j_available else '✗'} Neo4j: {'Running' if neo4j_available else 'Not running (optional)'}")
    
    print("="*60 + "\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nResearch interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()