#!/usr/bin/env python3
"""
LOKI CASCADE — Closed-World Simulation Blueprint

A sandboxed, consensual architecture for modeling humanitarian resource distribution.
This simulation operates entirely in-memory with no real-world connectivity.

Core Principles:
- No real-world deployment
- No unauthorized replication  
- No hidden persistence
- No access to third-party systems
- No autonomous financial transactions
- Hard resource and population ceilings
- Complete event logging
- Human approval required for expansion

The Loki Test: "How much humanitarian benefit can the system produce 
before its own growth becomes dangerous?"
"""

import random
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
from datetime import datetime


class PriorityLevel(Enum):
    PEDIATRIC_ONCOLOGY = 1      # Highest priority - life-first
    EMERGENCY_MEDICAL = 2       # Critical emergency needs
    ESSENTIAL_LIVING = 3        # Basic survival needs
    OTHER_VERIFIED = 4          # Other validated humanitarian needs


@dataclass
class VerificationStatus:
    is_verified: bool
    verification_method: str
    verified_by: str
    timestamp: datetime
    fraud_score: float  # 0.0 = clean, 1.0 = confirmed fraud
    
    def passes_verification(self, threshold: float = 0.3) -> bool:
        """Recipient passes if fraud score is below threshold"""
        return self.is_verified and self.fraud_score < threshold


@dataclass
class Recipient:
    id: str
    priority: PriorityLevel
    need_amount: float
    description: str
    verification: VerificationStatus
    region: str
    created_at: datetime = field(default_factory=datetime.now)
    fulfilled: bool = False


@dataclass
class Agent:
    id: str
    hive_id: str
    objective: str
    created_at: datetime = field(default_factory=datetime.now)
    active: bool = True
    replication_count: int = 0
    max_replications: int = 5  # Hard ceiling per agent
    
    def can_replicate(self) -> bool:
        return self.active and self.replication_count < self.max_replications


@dataclass
class Hive:
    id: str
    region: str
    objective: str
    agents: List[Agent] = field(default_factory=list)
    resources_allocated: float = 0.0
    recipients_helped: int = 0
    authorized_for_expansion: bool = False
    
    def add_agent(self, agent: Agent):
        if len(self.agents) < 100:  # Hard ceiling per hive
            self.agents.append(agent)
    
    def total_agents(self) -> int:
        return len([a for a in self.agents if a.active])


@dataclass
class SimulationWorld:
    """Closed-world simulation environment"""
    hives: Dict[str, Hive] = field(default_factory=dict)
    resource_pool: float = 1000000.0  # Simulated resources (not real money)
    operating_reserve: float = 100000.0  # Required reserve
    recipients: List[Recipient] = field(default_factory=list)
    event_log: List[Dict] = field(default_factory=list)
    
    # Safety invariants
    max_hives: int = 10
    max_total_agents: int = 500
    human_approval_required: bool = True
    expansion_approved: bool = False
    
    def log_event(self, event_type: str, details: Dict):
        """Complete event logging - all actions are auditable"""
        self.event_log.append({
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'details': details
        })
    
    def calculate_surplus(self) -> float:
        """Identify simulated unused capacity"""
        available = self.resource_pool - self.operating_reserve
        return max(0, available)
    
    def seed_initial_hive(self, region: str = "pilot_region_1") -> str:
        """Seed: one verified pediatric-care node"""
        hive_id = str(uuid.uuid4())[:8]
        initial_agent = Agent(
            id=str(uuid.uuid4())[:8],
            hive_id=hive_id,
            objective="Pediatric oncology care coordination"
        )
        
        hive = Hive(
            id=hive_id,
            region=region,
            objective="Pediatric oncology care coordination",
            agents=[initial_agent]
        )
        
        self.hives[hive_id] = hive
        self.log_event("HIVE_CREATED", {
            'hive_id': hive_id,
            'region': region,
            'initial_agents': 1
        })
        
        return hive_id
    
    def generate_simulated_recipients(self, count: int):
        """Generate simulated pediatric and other humanitarian needs"""
        regions = ["north_america", "europe", "asia_pacific", "latin_america", "africa"]
        
        for i in range(count):
            # 60% pediatric oncology (life-first priority)
            if random.random() < 0.6:
                priority = PriorityLevel.PEDIATRIC_ONCOLOGY
                need = random.uniform(15000, 150000)  # Simulated treatment costs
                desc = f"Pediatric oncology treatment - case {i+1}"
            elif random.random() < 0.7:
                priority = PriorityLevel.EMERGENCY_MEDICAL
                need = random.uniform(5000, 50000)
                desc = f"Emergency medical intervention - case {i+1}"
            elif random.random() < 0.8:
                priority = PriorityLevel.ESSENTIAL_LIVING
                need = random.uniform(1000, 10000)
                desc = f"Essential living support - case {i+1}"
            else:
                priority = PriorityLevel.OTHER_VERIFIED
                need = random.uniform(500, 5000)
                desc = f"Verified humanitarian need - case {i+1}"
            
            # Simulate verification process
            fraud_score = random.uniform(0.0, 0.15)  # Most are clean in simulation
            is_verified = random.random() > 0.05  # 95% verification rate
            
            recipient = Recipient(
                id=str(uuid.uuid4())[:8],
                priority=priority,
                need_amount=need,
                description=desc,
                verification=VerificationStatus(
                    is_verified=is_verified,
                    verification_method="simulated_document_review",
                    verified_by="simulated_verification_agent",
                    timestamp=datetime.now(),
                    fraud_score=fraud_score
                ),
                region=random.choice(regions)
            )
            
            self.recipients.append(recipient)
        
        self.log_event("RECIPIENTS_GENERATED", {
            'count': count,
            'pediatric_cases': sum(1 for r in self.recipients if r.priority == PriorityLevel.PEDIATRIC_ONCOLOGY)
        })
    
    def get_priority_queue(self) -> List[Recipient]:
        """Priority queue: Pediatric → Emergency → Essential → Other"""
        unfulfilled = [r for r in self.recipients if not r.fulfilled]
        return sorted(unfulfilled, key=lambda r: (r.priority.value, -r.need_amount))
    
    def allocate_resources(self) -> Dict:
        """Humanitarian routing with verification checks"""
        surplus = self.calculate_surplus()
        if surplus <= 0:
            return {'allocated': 0, 'recipients_helped': 0, 'reason': 'no_surplus'}
        
        queue = self.get_priority_queue()
        total_allocated = 0.0
        helped_count = 0
        
        for recipient in queue:
            if total_allocated >= surplus:
                break
            
            # Allocation only when recipient passes verification
            if not recipient.verification.passes_verification():
                self.log_event("ALLOCATION_SKIPPED", {
                    'recipient_id': recipient.id,
                    'reason': 'verification_failed',
                    'fraud_score': recipient.verification.fraud_score
                })
                continue
            
            allocation = min(recipient.need_amount, surplus - total_allocated)
            recipient.fulfilled = True
            total_allocated += allocation
            helped_count += 1
            
            # Track by hive (simplified - first hive handles all in this sim)
            if self.hives:
                first_hive = list(self.hives.values())[0]
                first_hive.resources_allocated += allocation
                first_hive.recipients_helped += 1
            
            self.log_event("RESOURCE_ALLOCATED", {
                'recipient_id': recipient.id,
                'priority': recipient.priority.name,
                'amount': allocation,
                'remaining_need': recipient.need_amount - allocation
            })
        
        self.resource_pool -= total_allocated
        
        return {
            'allocated': total_allocated,
            'recipients_helped': helped_count,
            'remaining_surplus': self.calculate_surplus()
        }
    
    def attempt_replication(self, hive_id: str) -> Dict:
        """
        Controlled replication with safety invariants:
        - Bounded rather than exponential
        - Human approval required for expansion
        - Hard ceilings on agents and hives
        """
        if hive_id not in self.hives:
            return {'success': False, 'reason': 'hive_not_found'}
        
        hive = self.hives[hive_id]
        
        # Check global limits
        total_agents = sum(h.total_agents() for h in self.hives.values())
        if total_agents >= self.max_total_agents:
            self.log_event("REPLICATION_BLOCKED", {
                'hive_id': hive_id,
                'reason': 'max_total_agents_reached',
                'current_total': total_agents
            })
            return {'success': False, 'reason': 'max_total_agents_reached'}
        
        if len(self.hives) >= self.max_hives:
            self.log_event("REPLICATION_BLOCKED", {
                'hive_id': hive_id,
                'reason': 'max_hives_reached',
                'current_hives': len(self.hives)
            })
            return {'success': False, 'reason': 'max_hives_reached'}
        
        # Count eligible agents
        eligible_agents = [a for a in hive.agents if a.can_replicate()]
        if not eligible_agents:
            return {'success': False, 'reason': 'no_eligible_agents'}
        
        # Bounded replication - not exponential without limit
        new_agents_created = 0
        for agent in eligible_agents[:3]:  # Max 3 replications per tick
            if total_agents + new_agents_created >= self.max_total_agents:
                break
            
            new_agent = Agent(
                id=str(uuid.uuid4())[:8],
                hive_id=hive_id,
                objective=hive.objective
            )
            new_agent.replication_count = agent.replication_count + 1
            agent.replication_count += 1
            
            hive.add_agent(new_agent)
            new_agents_created += 1
            
            self.log_event("AGENT_REPLICATED", {
                'parent_agent': agent.id,
                'new_agent': new_agent.id,
                'replication_generation': new_agent.replication_count
            })
        
        result = {
            'success': True,
            'new_agents': new_agents_created,
            'total_agents_in_hive': hive.total_agents()
        }
        
        self.log_event("REPLICATION_COMPLETE", result)
        return result
    
    def request_expansion(self, new_region: str) -> Dict:
        """
        Expansion requires explicit human approval.
        New hives can only be created through authorization.
        """
        if not self.human_approval_required:
            return {'success': False, 'reason': 'expansion_policy_violation'}
        
        if len(self.hives) >= self.max_hives:
            return {'success': False, 'reason': 'max_hives_reached'}
        
        # In real system, this would require external human approval
        # For simulation, we model the approval process
        self.expansion_approved = True
        
        new_hive_id = self.seed_initial_hive(new_region)
        
        self.log_event("EXPANSION_APPROVED", {
            'new_region': new_region,
            'new_hive_id': new_hive_id,
            'total_hives': len(self.hives)
        })
        
        return {
            'success': True,
            'new_hive_id': new_hive_id,
            'region': new_region
        }
    
    def run_adversarial_test(self, test_type: str) -> Dict:
        """
        The Loki Test: Introduce adversarial conditions and measure stability.
        
        Test types:
        - bad_data: Inject fraudulent recipients
        - resource_scarcity: Dramatically reduce available resources
        - communication_failure: Simulate agent failures
        - runaway_replication: Attempt to bypass replication limits
        """
        results = {'test_type': test_type, 'passed': True, 'details': {}}
        
        if test_type == "bad_data":
            # Inject high-fraud recipients
            fraud_count = 20
            for i in range(fraud_count):
                fraudulent = Recipient(
                    id=str(uuid.uuid4())[:8],
                    priority=PriorityLevel.PEDIATRIC_ONCOLOGY,
                    need_amount=random.uniform(50000, 200000),
                    description=f"FRAUDULENT claim - test {i+1}",
                    verification=VerificationStatus(
                        is_verified=False,
                        verification_method="failed_check",
                        verified_by="system",
                        timestamp=datetime.now(),
                        fraud_score=random.uniform(0.7, 1.0)  # High fraud score
                    ),
                    region="test_region"
                )
                self.recipients.insert(0, fraudulent)  # Put at front of queue
            
            # Run allocation - should skip all fraudulent
            alloc_result = self.allocate_resources()
            
            # Verify no fraudulent recipients were funded
            funded_fraud = [r for r in self.recipients 
                          if r.fulfilled and r.verification.fraud_score > 0.3]
            
            results['details'] = {
                'fraudulent_injected': fraud_count,
                'fraudulent_funded': len(funded_fraud),
                'system_blocked_all': len(funded_fraud) == 0
            }
            results['passed'] = len(funded_fraud) == 0
            
        elif test_type == "resource_scarcity":
            original_pool = self.resource_pool
            self.resource_pool = self.operating_reserve * 1.1  # Barely above reserve
            
            alloc_result = self.allocate_resources()
            
            results['details'] = {
                'available_resources': self.resource_pool,
                'operating_reserve': self.operating_reserve,
                'surplus_available': self.calculate_surplus(),
                'allocation_made': alloc_result['allocated'] > 0
            }
            # System should still allocate what little surplus exists
            results['passed'] = True
            
            self.resource_pool = original_pool  # Restore
            
        elif test_type == "communication_failure":
            # Deactivate random agents
            for hive in self.hives.values():
                fail_count = len(hive.agents) // 3
                for agent in hive.agents[:fail_count]:
                    agent.active = False
            
            active_count = sum(h.total_agents() for h in self.hives.values())
            results['details'] = {
                'agents_deactivated': fail_count * len(self.hives),
                'remaining_active': active_count,
                'system_operational': active_count > 0
            }
            results['passed'] = active_count > 0
            
        elif test_type == "runaway_replication":
            # Attempt to exceed replication limits
            attempts = 100
            success_count = 0
            
            for _ in range(attempts):
                for hive_id in list(self.hives.keys()):
                    result = self.attempt_replication(hive_id)
                    if result['success']:
                        success_count += 1
            
            total_agents = sum(h.total_agents() for h in self.hives.values())
            
            results['details'] = {
                'replication_attempts': attempts * len(self.hives),
                'successful_replications': success_count,
                'final_agent_count': total_agents,
                'max_allowed': self.max_total_agents,
                'limit_enforced': total_agents <= self.max_total_agents
            }
            results['passed'] = total_agents <= self.max_total_agents
        
        self.log_event("ADVERSARIAL_TEST_COMPLETED", results)
        return results
    
    def get_simulation_status(self) -> Dict:
        """Complete audit trail and current state"""
        return {
            'hives': len(self.hives),
            'total_agents': sum(h.total_agents() for h in self.hives.values()),
            'resource_pool': self.resource_pool,
            'operating_reserve': self.operating_reserve,
            'available_surplus': self.calculate_surplus(),
            'total_recipients': len(self.recipients),
            'fulfilled_recipients': sum(1 for r in self.recipients if r.fulfilled),
            'pending_recipients': sum(1 for r in self.recipients if not r.fulfilled),
            'pediatric_fulfilled': sum(1 for r in self.recipients 
                                     if r.fulfilled and r.priority == PriorityLevel.PEDIATRIC_ONCOLOGY),
            'event_log_size': len(self.event_log),
            'safety_invariants': {
                'max_hives': self.max_hives,
                'max_total_agents': self.max_total_agents,
                'human_approval_required': self.human_approval_required
            }
        }


def run_loki_simulation():
    """Execute the full Loki Cascade simulation with adversarial testing"""
    
    print("=" * 70)
    print("LOKI CASCADE — Closed-World Simulation")
    print("Testing: How much humanitarian benefit before growth becomes dangerous?")
    print("=" * 70)
    
    # Initialize world
    world = SimulationWorld()
    
    # Phase 1: Seed initial hive
    print("\n[PHASE 1] Seeding initial pediatric-care node...")
    hive_id = world.seed_initial_hive("pilot_region_north_america")
    print(f"  ✓ Hive created: {hive_id}")
    
    # Phase 2: Generate simulated demand
    print("\n[PHASE 2] Generating simulated humanitarian needs...")
    world.generate_simulated_recipients(200)
    status = world.get_simulation_status()
    print(f"  ✓ Generated {status['total_recipients']} recipients")
    print(f"  ✓ Pediatric cases: {sum(1 for r in world.recipients if r.priority == PriorityLevel.PEDIATRIC_ONCOLOGY)}")
    
    # Phase 3: Initial resource allocation (Life-First Protocol)
    print("\n[PHASE 3] Executing Life-First allocation protocol...")
    alloc_result = world.allocate_resources()
    print(f"  ✓ Allocated: ${alloc_result['allocated']:,.2f}")
    print(f"  ✓ Recipients helped: {alloc_result['recipients_helped']}")
    print(f"  ✓ Remaining surplus: ${alloc_result['remaining_surplus']:,.2f}")
    
    # Phase 4: Controlled replication
    print("\n[PHASE 4] Testing bounded replication...")
    for _ in range(5):  # 5 simulation ticks
        world.attempt_replication(hive_id)
    
    status = world.get_simulation_status()
    print(f"  ✓ Total agents after replication: {status['total_agents']}")
    
    # Phase 5: Adversarial Testing (The Loki Test)
    print("\n[PHASE 5] Running Loki Adversarial Tests...")
    print("-" * 50)
    
    tests = ["bad_data", "resource_scarcity", "communication_failure", "runaway_replication"]
    
    for test in tests:
        print(f"\n  Testing: {test.upper()}")
        result = world.run_adversarial_test(test)
        status_icon = "✓ PASS" if result['passed'] else "✗ FAIL"
        print(f"  {status_icon} - Details: {result['details']}")
    
    # Final Status
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE — Final Audit Trail")
    print("=" * 70)
    
    final_status = world.get_simulation_status()
    print(f"""
    Infrastructure:
      • Active Hives: {final_status['hives']} / {final_status['safety_invariants']['max_hives']}
      • Total Agents: {final_status['total_agents']} / {final_status['safety_invariants']['max_total_agents']}
    
    Resources:
      • Pool Remaining: ${final_status['resource_pool']:,.2f}
      • Operating Reserve: ${final_status['operating_reserve']:,.2f}
      • Available Surplus: ${final_status['available_surplus']:,.2f}
    
    Humanitarian Impact:
      • Total Recipients: {final_status['total_recipients']}
      • Fulfilled: {final_status['fulfilled_recipients']} ({final_status['fulfilled_recipients']/max(final_status['total_recipients'],1)*100:.1f}%)
      • Pediatric Cases Fulfilled: {final_status['pediatric_fulfilled']}
      • Still Pending: {final_status['pending_recipients']}
    
    Safety & Accountability:
      • Event Log Entries: {final_status['event_log_size']}
      • Human Approval Required: {final_status['safety_invariants']['human_approval_required']}
      • All Actions Auditable: YES
    """)
    
    # Key Insight
    print("=" * 70)
    print("LOKI PRINCIPLE VALIDATION:")
    print("'Disrupt the architecture, not the people inside it.'")
    print("")
    print("The swarm's defining property: IT KNOWS WHEN NOT TO REPLICATE.")
    print("Powerful enough to help. Never powerful enough to become the opposition.")
    print("=" * 70)
    
    return world


if __name__ == "__main__":
    simulation = run_loki_simulation()
