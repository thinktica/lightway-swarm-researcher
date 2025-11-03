#!/usr/bin/env python3
"""
Challenge-Driven Swarm Intelligence Research Agent
===================================================
Updated for Thinktica's structured agent system with ABC interfaces
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
from collections import defaultdict, deque
import time
import re
from pathlib import Path

# Import from Thinktica's new system
from thinktica import ResearchAgent
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Check for optional dependencies
EMBEDDINGS_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
LLAMACPP_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

try:
    from llama_cpp import Llama
    LLAMACPP_AVAILABLE = True
except ImportError:
    pass

try:
    import arxiv
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "arxiv"], capture_output=True)
    import arxiv


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
    
    def __init__(self, parent_agent=None):
        self.backend = None
        self.model = None
        self.parent_agent = parent_agent
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the best available backend"""
        if self._try_llamacpp():
            self.backend = 'llamacpp'
            if self.parent_agent:
                self.parent_agent.emit("Using Llama.cpp backend", type="system")
            return
        
        if self._try_transformers():
            self.backend = 'transformers'
            if self.parent_agent:
                self.parent_agent.emit("Using Transformers backend", type="system")
            return
        
        if self._try_ollama():
            self.backend = 'ollama'
            if self.parent_agent:
                self.parent_agent.emit("Using Ollama backend", type="system")
            return
        
        self.backend = 'mock'
        if self.parent_agent:
            self.parent_agent.emit("No local LLM available - using mock responses", type="warning")
    
    def _try_llamacpp(self) -> bool:
        """Try to use llama.cpp"""
        if not LLAMACPP_AVAILABLE:
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "llama-cpp-python"],
                    capture_output=True,
                    timeout=60
                )
                from llama_cpp import Llama
            except:
                return False
        
        try:
            from llama_cpp import Llama
            model_dir = Path.home() / ".cache" / "llm_models"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
            
            if not model_path.exists():
                if self.parent_agent:
                    self.parent_agent.emit("Downloading Mistral 7B model...", type="progress")
                
                url = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
                
                try:
                    import urllib.request
                    urllib.request.urlretrieve(url, model_path)
                    if self.parent_agent:
                        self.parent_agent.emit(f"Model downloaded to {model_path}", type="system")
                except Exception as e:
                    return False
            
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
                return True
                
        except Exception as e:
            return False
        
        return False
    
    def _try_transformers(self) -> bool:
        """Try Transformers backend"""
        if not TRANSFORMERS_AVAILABLE:
            return False
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            
            model_name = "microsoft/phi-2"
            
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
            pass
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


class LocalSwarmAgent:
    """Base class for local swarm agents"""
    
    def __init__(self, role: SwarmRole, llm_backend=None):
        self.role = role
        self.processing_count = 0
        self.llm_backend = llm_backend
    
    def process(self, prompt: str) -> str:
        """Process with local LLM"""
        self.processing_count += 1
        
        if not self.llm_backend:
            return f"[{self.role.value} - no LLM backend]"
        
        role_prompt = f"You are a {self.role.value} agent. {prompt}"
        response = self.llm_backend.generate(role_prompt)
        
        if response and not response.startswith('['):
            return response
        
        return f"[{self.role.value} processing failed]"


class ChallengeExtractorAgent(LocalSwarmAgent):
    """Extract challenges from text"""
    
    def __init__(self, llm_backend=None):
        super().__init__(SwarmRole.CHALLENGE_EXTRACTOR, llm_backend)
    
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
    
    def __init__(self, llm_backend=None):
        super().__init__(SwarmRole.CHALLENGE_SPECIFIER, llm_backend)
    
    def specify_challenge(self, challenge: Challenge, context: SwarmContext) -> List[Challenge]:
        """Break down challenge into specific sub-challenges"""
        prompt = f"""
        Break down this challenge into SPECIFIC sub-problems:
        
        Challenge: {challenge.statement}
        Type: {challenge.type}
        Context: {challenge.parent_context}
        
        Generate 3-5 concrete sub-challenges that:
        1. Are more specific than the parent
        2. Can be individually researched
        3. Together would solve the parent challenge
        
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
    
    def __init__(self, llm_backend=None):
        super().__init__(SwarmRole.RESOLUTION_AGENT, llm_backend)
    
    def attempt_resolution(self, challenge: Challenge, findings: List[Dict], context: SwarmContext) -> Dict:
        """Try to resolve challenge with available information"""
        findings_text = "\n".join([f"- {f.get('statement', '')}" for f in findings[:10]])
        
        prompt = f"""
        Attempt to resolve this challenge using available findings:
        
        Challenge: {challenge.statement}
        Type: {challenge.type}
        
        Available findings:
        {findings_text}
        
        Determine:
        1. Can this be resolved with current knowledge?
        2. If yes, what's the solution?
        3. If no, what specific information is missing?
        4. Confidence in resolution (0-1)
        
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
    
    def __init__(self, llm_backend=None):
        super().__init__(SwarmRole.ANALYZER, llm_backend)
    
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
            "contradictions": []
        }


class ExtractorAgent(LocalSwarmAgent):
    """Extract findings from text"""
    
    def __init__(self, llm_backend=None):
        super().__init__(SwarmRole.EXTRACTOR, llm_backend)
    
    def extract_findings(self, response: str, analysis: Dict) -> List[Dict]:
        """Extract concrete findings"""
        prompt = f"""
        Extract concrete findings from this text:
        {response[:1000]}
        
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
    
    def __init__(self, llm_backend=None):
        super().__init__(SwarmRole.COMPARATOR, llm_backend)
    
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
            "novelty": 0.5
        }


class ChallengerAgent(LocalSwarmAgent):
    """Generate challenges from findings"""
    
    def __init__(self, llm_backend=None):
        super().__init__(SwarmRole.CHALLENGER, llm_backend)
    
    def generate_challenges(self, findings: List[Dict], context: SwarmContext) -> List[str]:
        """Generate challenging questions"""
        findings_summary = "\n".join([f.get('statement', '')[:100] for f in findings[:3]])
        
        prompt = f"""
        Generate challenging questions that:
        1. Test validity of findings
        2. Identify edge cases
        3. Expose assumptions
        
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
    
    def __init__(self, remote_llm=None, llm_backend=None):
        super().__init__(SwarmRole.EVALUATOR, llm_backend)
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
        
        return evaluation


class HypothesisAgent(LocalSwarmAgent):
    """Generate hypotheses"""
    
    def __init__(self, llm_backend=None):
        super().__init__(SwarmRole.HYPOTHESIS, llm_backend)
    
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
        
        Generate 3 testable hypotheses.
        
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
    
    def __init__(self, neo4j_driver=None, remote_llm=None, parent_agent=None):
        self.driver = neo4j_driver
        self.remote_llm = remote_llm
        self.parent_agent = parent_agent
        
        # Initialize LLM backend for all agents
        self.llm_backend = LocalLLMBackend(parent_agent)
        
        # Initialize all agents with shared LLM backend
        self.analyzer = AnalyzerAgent(self.llm_backend)
        self.extractor = ExtractorAgent(self.llm_backend)
        self.comparator = ComparatorAgent(self.llm_backend)
        self.challenger = ChallengerAgent(self.llm_backend)
        self.evaluator = EvaluatorAgent(remote_llm, self.llm_backend)
        self.hypothesis = HypothesisAgent(self.llm_backend)
        
        # New challenge agents
        self.challenge_extractor = ChallengeExtractorAgent(self.llm_backend)
        self.challenge_specifier = ChallengeSpecifierAgent(self.llm_backend)
        self.resolution_agent = ResolutionAgent(self.llm_backend)
        
        # Research tree
        self.research_tree: Dict[str, ResearchNode] = {}
        self.challenges_tree: Dict[str, Challenge] = {}
        self.current_node_id: Optional[str] = None
        
        if self.parent_agent:
            self.parent_agent.emit("Swarm orchestrator initialized", type="system")
    
    async def process_remote_response(self, 
                                     response: str, 
                                     context: SwarmContext) -> Dict:
        """Process response with challenge-driven approach"""
        
        if self.parent_agent:
            self.parent_agent.emit(f"Processing iteration {context.iteration}", type="progress")
        
        iteration_results = {
            'iteration': context.iteration,
            'findings': [],
            'challenges': [],
            'resolutions': [],
            'quality': 0.0,
            'decision': 'continue'
        }
        
        # Phase 1: Analyze response
        analysis = self.analyzer.analyze_response(response, context)
        
        # Phase 2: Extract findings
        findings = self.extractor.extract_findings(response, analysis)
        context.findings_buffer.extend(findings)
        iteration_results['findings'] = findings
        
        # Phase 3: Extract challenges
        new_challenges = self.challenge_extractor.extract_challenges(response, context)
        context.challenges_buffer.extend(new_challenges)
        iteration_results['challenges'] = new_challenges
        
        # Phase 4: Attempt to resolve existing challenges
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
                    if self.parent_agent:
                        self.parent_agent.emit(
                            f"Resolved challenge: {challenge.statement[:50]}",
                            type="resolution",
                            confidence=resolution['confidence']
                        )
                elif resolution['confidence'] > 0.3:
                    challenge.status = NodeStatus.PARTIALLY_RESOLVED
                    challenge.resolution_attempts.append(resolution)
        
        iteration_results['resolutions'] = resolutions
        
        # Phase 5: Specify unresolved challenges
        for challenge in context.challenges_buffer:
            if challenge.status == NodeStatus.OPEN and len(challenge.sub_challenges) == 0:
                sub_challenges = self.challenge_specifier.specify_challenge(challenge, context)
                challenge.sub_challenges = [sc.id for sc in sub_challenges]
                context.challenges_buffer.extend(sub_challenges)
        
        # Phase 6: Compare to goals
        comparison = self.comparator.compare_to_goals(findings, context)
        
        # Phase 7: Generate new challenges from findings
        challenge_questions = self.challenger.generate_challenges(findings, context)
        
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
        evaluation = self.evaluator.evaluate_iteration(context)
        context.quality_scores.append(evaluation['quality_score'])
        
        # Phase 9: Store to Neo4j if quality threshold met
        if evaluation['ready_for_storage'] and self.driver:
            self._store_iteration_data(context, evaluation)
        
        # Phase 10: Generate hypotheses if appropriate
        if evaluation.get('generate_hypothesis', False):
            new_hypotheses = self.hypothesis.generate_hypotheses(context)
            iteration_results['hypotheses'] = new_hypotheses
        
        iteration_results['quality'] = evaluation['quality_score']
        iteration_results['decision'] = evaluation['decision']
        iteration_results['comparison'] = comparison
        
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
                
                # Store challenges
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
                            
                            stored_count += 1
                            
                        except Exception as e:
                            logger.error(f"Failed to store finding: {e}")
                
                if self.parent_agent:
                    self.parent_agent.emit(
                        f"Stored {stored_count} items to Neo4j",
                        type="progress"
                    )
                
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
                if self.parent_agent:
                    self.parent_agent.emit(
                        f"Quality threshold reached at iteration {i+1}",
                        type="progress"
                    )
                break
            
            # Refine response based on unresolved challenges
            unresolved = [c for c in context.challenges_buffer if c.status == NodeStatus.OPEN]
            if unresolved:
                challenge_text = "\n".join([f"- {c.statement}" for c in unresolved[:5]])
                response = response + f"\n\nUnresolved challenges:\n{challenge_text}"
        
        # Calculate resolution rate
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
    
    def __init__(self, provider: str = "groq", api_key: Optional[str] = None, parent_agent=None):
        self.provider = provider
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        self.parent_agent = parent_agent
        
        if not self.api_key:
            msg = f"No API key for {provider}. Set {provider.upper()}_API_KEY"
            if self.parent_agent:
                self.parent_agent.emit(msg, type="warning")
            else:
                logger.warning(msg)
    
    def get_strategic_direction(self, goal: str, current_state: Dict) -> str:
        """Get strategic direction from remote LLM"""
        
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
                    
                    if self.parent_agent:
                        self.parent_agent.emit(
                            f"Rate limit hit. Waiting {wait_time} seconds...",
                            type="warning"
                        )
                    
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                        continue
                    else:
                        return "[Rate limit exceeded]"
                else:
                    return "[Remote LLM error]"
            
        except Exception as e:
            if self.parent_agent:
                self.parent_agent.emit(f"Groq error: {e}", type="error")
            return "[Remote LLM error]"


class Agent(ResearchAgent):
    """
    Challenge-Driven Research Agent for Thinktica.
    
    Inherits from ResearchAgent to get:
    - Automatic context injection
    - Event emission through self.emit()
    - Neo4j connection management
    """
    
    def __init__(self):
        """Initialize the agent"""
        super().__init__()
        
        self.emit("Initializing Challenge-Driven Research Agent", type="system")
        
        # Initialize components
        self.api_call_count = 0
        self.last_api_call = None
        
        # Setup LLM backend
        self.llm_backend = LocalLLMBackend(self)
        
        # Setup remote LLM
        self.remote_llm = RemoteLLMOrchestrator(
            provider=os.getenv('LLM_PROVIDER', 'groq'),
            parent_agent=self
        )
        
        # Connect to Neo4j using injected credentials
        self.driver = self._connect_neo4j() if self.has_neo4j else None
        
        # Initialize swarm
        self.swarm = SwarmOrchestrator(self.driver, self.remote_llm, self)
        
        # Research state
        self.tree_root: Optional[ResearchNode] = None
        self.challenge_queue = deque()
        self.node_queue = deque()
        
        # ArXiv client
        try:
            self.arxiv_client = arxiv.Client()
        except:
            self.arxiv_client = None
        
        self.emit(
            "Agent ready",
            type="system",
            workspace=self.workspace,
            investigation_id=self.investigation_id,
            has_neo4j=self.has_neo4j
        )
    
    def _connect_neo4j(self) -> Optional[GraphDatabase.driver]:
        """Connect to Neo4j using injected credentials"""
        try:
            driver = GraphDatabase.driver(
                self.neo4j_url,
                auth=(self.neo4j_user, self.neo4j_pass)
            )
            with driver.session() as session:
                session.run("RETURN 1")
            
            self.emit("Connected to Neo4j", type="system")
            return driver
            
        except Exception as e:
            self.emit(f"Neo4j connection failed: {str(e)[:100]}", type="warning")
            return None
    
    def research(self, question: str) -> Dict[str, Any]:
        """
        Main research method required by IResearchable.
        
        Args:
            question: Research question to investigate
            
        Returns:
            Dictionary containing research results
        """
        self.emit(
            f"Starting research: {question}",
            type="progress",
            question=question,
            persist=True
        )
        
        # Run async research
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._research_async(question, max_depth=3))
            
            # Collect results
            all_findings = []
            all_challenges = []
            
            if hasattr(self.swarm, 'research_tree'):
                for node in self.swarm.research_tree.values():
                    all_findings.extend(node.findings)
                    all_challenges.extend(node.challenges)
            
            # Sort by confidence
            all_findings.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            # Calculate statistics
            total_nodes = len(self.swarm.research_tree) if hasattr(self.swarm, 'research_tree') else 0
            resolved_nodes = len([
                n for n in self.swarm.research_tree.values() 
                if n.status == NodeStatus.RESOLVED
            ]) if hasattr(self.swarm, 'research_tree') else 0
            
            result = {
                "question": question,
                "findings": all_findings[:20],
                "challenges": all_challenges[:10],
                "nodes_explored": total_nodes,
                "nodes_resolved": resolved_nodes,
                "resolution_rate": resolved_nodes / max(total_nodes, 1),
                "confidence": np.mean([f.get('confidence', 0) for f in all_findings[:10]]) if all_findings else 0,
                "workspace": self.workspace,
                "investigation_id": self.investigation_id
            }
            
            self.emit(
                "Research complete",
                type="discovery",
                confidence=result['confidence'],
                findings_count=len(all_findings),
                challenges_count=len(all_challenges),
                resolution_rate=result['resolution_rate'],
                persist=True
            )
            
            # Store top findings to Neo4j
            if self.driver and self.investigation_id:
                self._store_findings(all_findings[:10])
            
            return result
            
        finally:
            loop.close()
    
    def validate(self, finding: Dict[str, Any]) -> float:
        """Validate a finding"""
        confidence = finding.get('confidence', 0.5)
        
        if 'evidence' in finding and finding['evidence']:
            confidence += 0.1
        if 'sources' in finding and len(finding.get('sources', [])) > 2:
            confidence += 0.1
        if 'peer_reviewed' in finding and finding['peer_reviewed']:
            confidence += 0.2
        
        confidence = min(confidence, 1.0)
        
        self.emit(
            f"Validated finding",
            type="validation",
            confidence=confidence
        )
        
        return confidence
    
    def query(self, cypher: str) -> List[Dict[str, Any]]:
        """Execute Cypher query"""
        if not self.driver:
            self.emit("Neo4j not available", type="warning")
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run(cypher)
                records = [dict(record) for record in result]
                
                self.emit(
                    f"Query returned {len(records)} results",
                    type="query"
                )
                
                return records
                
        except Exception as e:
            self.emit(f"Query error: {str(e)}", type="error")
            return []
    
    def schema(self) -> Dict[str, Any]:
        """Return knowledge graph schema"""
        return {
            "nodes": [
                "ResearchNode",
                "Challenge",
                "Discovery",
                "Resolution",
                "Investigation",
                "Finding"
            ],
            "relationships": [
                {"type": "REFINES", "from": "ResearchNode", "to": "ResearchNode"},
                {"type": "BLOCKS", "from": "Challenge", "to": "ResearchNode"},
                {"type": "RESOLVES", "from": "Resolution", "to": "Challenge"},
                {"type": "DISCOVERED", "from": "Investigation", "to": "Finding"}
            ]
        }
    
    async def _research_async(self, goal: str, max_depth: int = 3):
        """Internal async research method"""
        self.emit("Starting challenge-driven research", type="progress")
        
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
            
            self.emit(
                f"Exploring: {current_node.content[:80]}",
                type="progress",
                node_type=current_node.type.value,
                depth=current_node.depth
            )
            
            # Get strategic direction
            current_state = {
                'explored_count': nodes_explored,
                'current_depth': current_node.depth,
                'resolution_rate': challenges_resolved / max(nodes_explored, 1),
                'challenges': []
            }
            
            # Rate limiting
            if self.last_api_call:
                elapsed = time.time() - self.last_api_call
                if elapsed < 2.4:
                    await asyncio.sleep(2.4 - elapsed)
            
            self.last_api_call = time.time()
            
            strategic_response = self.remote_llm.get_strategic_direction(
                current_node.content, current_state
            )
            
            # Fetch external knowledge
            papers = await self.fetch_external_knowledge(current_node.content)
            
            # Process through swarm
            swarm_results = await self.swarm.run_swarm_loop(
                response=strategic_response + "\n\nPapers:\n" + json.dumps(papers),
                node=current_node,
                global_goal=goal,
                max_iterations=5
            )
            
            # Update node
            current_node.findings = swarm_results['findings']
            current_node.challenges = swarm_results['challenges']
            current_node.confidence = swarm_results['final_quality']
            
            # Check if resolved
            if swarm_results['resolution_rate'] > 0.7:
                current_node.status = NodeStatus.RESOLVED
                current_node.resolved_at = datetime.now()
                challenges_resolved += swarm_results['resolved_challenges']
                
                self.emit(
                    f"Resolved: {current_node.content[:50]}",
                    type="resolution",
                    confidence=current_node.confidence
                )
            else:
                current_node.status = NodeStatus.PARTIALLY_RESOLVED
            
            # Create child nodes
            unresolved = [c for c in swarm_results['challenges'] if c['status'] == 'open']
            for challenge in unresolved[:3]:
                child = self.create_research_node(
                    content=challenge['statement'],
                    node_type=NodeType.SUB_CHALLENGE,
                    parent=current_node
                )
                self.node_queue.append(child)
        
        self.emit(
            "Research exploration complete",
            type="progress",
            nodes_explored=nodes_explored,
            challenges_resolved=challenges_resolved
        )
    
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
        
        self.emit(
            f"Created node: {content[:50]}",
            type="progress",
            node_type=node_type.value
        )
        
        return node
    
    async def fetch_external_knowledge(self, query: str) -> List[Dict]:
        """Fetch from ArXiv"""
        if not self.arxiv_client:
            return []
        
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
            
            self.emit(f"Found {len(papers)} papers", type="progress")
            
        except Exception as e:
            self.emit(f"ArXiv error: {e}", type="error")
        
        return papers
    
    def _store_findings(self, findings: List[Dict]):
        """Store top findings to Neo4j linked to investigation"""
        if not self.driver or not self.investigation_id:
            return
        
        try:
            with self.driver.session() as session:
                stored = 0
                for finding in findings:
                    if finding.get('confidence', 0) > 0.5:
                        session.run("""
                            MATCH (i:Investigation {id: $inv_id})
                            CREATE (f:Finding {
                                statement: $statement,
                                confidence: $confidence,
                                timestamp: datetime()
                            })
                            CREATE (i)-[:DISCOVERED]->(f)
                        """, 
                        inv_id=self.investigation_id,
                        statement=finding.get('statement', ''),
                        confidence=finding.get('confidence', 0))
                        stored += 1
                
                self.emit(f"Stored {stored} findings", type="progress")
                
        except Exception as e:
            self.emit(f"Could not store findings: {str(e)[:50]}", type="error")
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
            self.emit("Closed Neo4j connection", type="system")


# Entry point for Thinktica
if __name__ == '__main__':
    agent = Agent()
    
    # Get research question from environment or default
    question = os.getenv('THINKTICA_RESEARCH_QUESTION', 'How to cure cancer?')
    
    # Run research
    results = agent.research(question)
    
    print("\n" + "="*60)
    print("RESEARCH RESULTS")
    print("="*60)
    print(f"Question: {results['question']}")
    print(f"Findings: {results.get('findings_count', len(results['findings']))}")
    print(f"Challenges: {len(results['challenges'])}")
    print(f"Resolution Rate: {results['resolution_rate']:.1%}")
    print(f"Confidence: {results['confidence']:.1%}")
    
    print("\nTop Findings:")
    for i, finding in enumerate(results['findings'][:5], 1):
        print(f"{i}. [{finding.get('confidence', 0):.1%}] {finding.get('statement', '')[:100]}")