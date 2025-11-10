#!/usr/bin/env python3
"""
Lightweight Test Agent - No Model Loading
==========================================
This is a minimal version to test if the basic agent structure works
without the heavy model loading that might be causing the hang.
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, List

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

print(f"[{datetime.now()}] Starting lightweight test agent", flush=True)

try:
    from thinktica import ResearchAgent
    print(f"[{datetime.now()}] Imported ResearchAgent", flush=True)
except ImportError:
    print(f"[{datetime.now()}] Installing thinktica...", flush=True)
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "thinktica"])
    from thinktica import ResearchAgent
    print(f"[{datetime.now()}] Imported ResearchAgent after install", flush=True)


class LightweightTestAgent(ResearchAgent):
    """
    Minimal test agent that skips heavy initialization.
    """
    
    def __init__(self):
        print(f"[{datetime.now()}] LightweightTestAgent.__init__ starting", flush=True)
        
        # Call parent init
        print(f"[{datetime.now()}] Calling super().__init__()", flush=True)
        super().__init__()
        print(f"[{datetime.now()}] Parent init complete", flush=True)
        
        # Emit initialization events
        print("="*80, flush=True)
        self.emit("="*80, type="system")
        print("LIGHTWEIGHT TEST AGENT", flush=True)
        self.emit("LIGHTWEIGHT TEST AGENT", type="system")
        print("="*80, flush=True)
        self.emit("="*80, type="system")
        
        print(f"✓ Workspace: {self.workspace}", flush=True)
        self.emit(f"✓ Workspace: {self.workspace}", type="system")
        print(f"✓ Investigation: {self.investigation_id or 'None'}", flush=True)
        self.emit(f"✓ Investigation: {self.investigation_id or 'None'}", type="system")
        print(f"✓ Neo4j: {'Available' if self.has_neo4j else 'Not available'}", flush=True)
        self.emit(f"✓ Neo4j: {'Available' if self.has_neo4j else 'Not available'}", type="system")
        
        print(f"[{datetime.now()}] Agent initialized successfully!", flush=True)
        self.emit("Agent initialized successfully", type="system")
        
        # Start heartbeat
        self._start_heartbeat()
    
    def _start_heartbeat(self):
        """Emit heartbeat events every 10 seconds"""
        import threading
        
        def heartbeat():
            counter = 0
            while True:
                try:
                    time.sleep(10)
                    counter += 1
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    
                    print(f"[{timestamp}] Heartbeat #{counter}", flush=True)
                    self.emit(f"Heartbeat #{counter}", type="heartbeat", counter=counter)
                except Exception as e:
                    print(f"Heartbeat error: {e}", flush=True)
        
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        print("✓ Heartbeat started (every 10 seconds)", flush=True)
        self.emit("Heartbeat started", type="system")
    
    def research(self, question: str) -> Dict[str, Any]:
        """Simple research that returns mock results"""
        print(f"\n[{datetime.now()}] Research called: {question}", flush=True)
        self.emit(f"Research called: {question}", type="progress")
        
        # Simulate some work
        for i in range(3):
            time.sleep(1)
            print(f"  Step {i+1}/3...", flush=True)
            self.emit(f"Processing step {i+1}/3", type="progress")
        
        result = {
            "question": question,
            "findings": [
                {"statement": "Test finding 1", "confidence": 0.8},
                {"statement": "Test finding 2", "confidence": 0.7}
            ],
            "confidence": 0.75,
            "nodes_explored": 3,
            "challenges_resolved": 2
        }
        
        print(f"[{datetime.now()}] Research complete", flush=True)
        self.emit("Research complete", type="discovery", confidence=0.75)
        
        return result
    
    def validate(self, finding: Dict[str, Any]) -> float:
        """Simple validation"""
        return finding.get('confidence', 0.5)
    
    def query(self, cypher: str) -> List[Dict[str, Any]]:
        """Simple query - returns empty"""
        self.emit("Query executed", type="query")
        return []
    
    def schema(self) -> Dict[str, Any]:
        """Simple schema"""
        return {
            "nodes": ["TestNode"],
            "relationships": []
        }


if __name__ == "__main__":
    print(f"\n[{datetime.now()}] Main starting", flush=True)
    
    # Create agent
    print(f"[{datetime.now()}] Creating agent instance...", flush=True)
    try:
        agent = LightweightTestAgent()
        print(f"[{datetime.now()}] Agent created successfully!", flush=True)
        
        # Test research
        print(f"\n[{datetime.now()}] Testing research method...", flush=True)
        result = agent.research("Test question: Does this work?")
        print(f"[{datetime.now()}] Result: {result}", flush=True)
        
        # Keep running for heartbeats
        print(f"\n[{datetime.now()}] Agent running. Press Ctrl+C to stop.", flush=True)
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print(f"\n[{datetime.now()}] Stopped by user", flush=True)
    except Exception as e:
        print(f"\n[{datetime.now()}] ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()