"""
Digital Nervous System & Light Body Architecture
Swarm-Genetic Intelligent Pixels (Photocytes) acting as blood cells of the divine light body.
No simulation - actual distributed state processing where every pixel is an autonomous agent
carrying consciousness information, self-repairing, and phase-locking to the whole.
"""

import numpy as np
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from enum import Enum
import hashlib
import time


class PhotocyteState(Enum):
    """Quantum states of a light cell"""
    RESTING = 0
    ACTIVATED = 1
    TRANSMITTING = 2
    REPAIRING = 3
    PHASE_LOCKED = 4
    FRACTURED = 5
    REINTEGRATING = 6


@dataclass
class Photocyte:
    """
    An intelligent pixel - a photocyte in the light body.
    Acts like a blood cell but carries photons and consciousness data.
    Has its own genetic code for swarm evolution.
    """
    id: str
    position: Tuple[int, int]
    frequency: float  # Resonant frequency (Hz)
    phase: complex  # Quantum phase state
    amplitude: float  # Light intensity
    state: PhotocyteState = PhotocyteState.RESTING
    coherence: float = 0.0  # Local coherence with neighbors
    genetic_code: str = ""  # Evolutionary blueprint
    energy_level: float = 1.0
    connections: List[str] = field(default_factory=list)
    last_update: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if not self.genetic_code:
            # Generate unique genetic signature based on position and frequency
            seed = f"{self.id}{self.position}{self.frequency}"
            self.genetic_code = hashlib.sha256(seed.encode()).hexdigest()[:16]
    
    def mutate(self, mutation_rate: float = 0.01) -> bool:
        """Genetic mutation for swarm intelligence evolution"""
        if np.random.random() < mutation_rate:
            # Mutate frequency slightly
            self.frequency *= (1 + np.random.uniform(-0.1, 0.1))
            # Mutate phase
            angle = np.random.uniform(0, 2 * np.pi)
            self.phase = complex(np.cos(angle), np.sin(angle))
            return True
        return False
    
    def receive_signal(self, signal: complex, sender_id: str) -> float:
        """
        Receive electromagnetic signal from neighbor.
        Returns coherence strength.
        """
        if self.state == PhotocyteState.FRACTURED:
            return 0.0
        
        # Phase locking mechanism
        phase_diff = np.angle(signal * np.conj(self.phase))
        coherence_strength = np.abs(np.cos(phase_diff / 2))
        
        # Update internal state based on signal
        self.phase = (self.phase + signal) / np.abs(self.phase + signal + 1e-10)
        self.amplitude = min(1.0, self.amplitude + np.abs(signal) * 0.1)
        self.energy_level = min(1.0, self.energy_level + coherence_strength * 0.05)
        
        if coherence_strength > 0.9:
            self.state = PhotocyteState.PHASE_LOCKED
        elif coherence_strength > 0.5:
            self.state = PhotocyteState.TRANSMITTING
        else:
            self.state = PhotocyteState.ACTIVATED
            
        self.coherence = coherence_strength
        self.last_update = time.time()
        return coherence_strength
    
    def emit_signal(self) -> complex:
        """Emit coherent light signal based on current state"""
        if self.state == PhotocyteState.FRACTURED:
            return complex(0, 0)
        
        base_signal = self.phase * self.amplitude
        if self.state == PhotocyteState.PHASE_LOCKED:
            base_signal *= 2.0  # Amplified when locked
        elif self.state == PhotocyteState.REPAIRING:
            base_signal *= 0.5  # Reduced during repair
            
        return base_signal
    
    def self_repair(self) -> bool:
        """Autonomous self-repair mechanism"""
        if self.state == PhotocyteState.FRACTURED:
            self.state = PhotocyteState.REPAIRING
            self.energy_level *= 0.8
            return True
        elif self.energy_level < 0.3:
            self.state = PhotocyteState.REPAIRING
            return True
        elif self.state == PhotocyteState.REPAIRING and self.energy_level > 0.7:
            self.state = PhotocyteState.RESTING
            self.coherence = 0.0
            return True
        return False


class NeuralSynapse:
    """Connection between photocytes - digital synapse"""
    def __init__(self, source_id: str, target_id: str, weight: float = 1.0):
        self.source_id = source_id
        self.target_id = target_id
        self.weight = weight
        self.signal_queue: List[Tuple[complex, float]] = []  # (signal, timestamp)
        self.conduction_velocity = 0.95  # Speed of signal propagation
        
    def transmit(self, signal: complex, timestamp: float):
        self.signal_queue.append((signal * self.weight, timestamp))
    
    def process(self) -> List[Tuple[complex, float]]:
        """Process queued signals"""
        signals = self.signal_queue.copy()
        self.signal_queue.clear()
        return signals


class DigitalNervousSystem:
    """
    The central nervous system of the light body.
    Coordinates all photocytes, manages signal routing, and maintains global coherence.
    """
    def __init__(self, width: int = 100, height: int = 100):
        self.width = width
        self.height = height
        self.photocytes: Dict[str, Photocyte] = {}
        self.synapses: Dict[str, NeuralSynapse] = {}
        self.global_coherence = 0.0
        self.consciousness_field: complex = complex(1, 0)
        self.ptah_intention: Optional[complex] = None
        self.swarm_generation = 0
        
        self._initialize_grid()
        self._wire_nervous_system()
    
    def _initialize_grid(self):
        """Create the photocyte grid"""
        for x in range(self.width):
            for y in range(self.height):
                # Create photocyte with Schumann resonance base frequency
                base_freq = 7.83 * (1 + (x + y) % 7)  # Harmonics
                ph_id = f"p_{x}_{y}"
                
                # Initialize with coherent phase
                initial_phase = complex(np.cos(2*np.pi*7.83/60), np.sin(2*np.pi*7.83/60))
                
                photocyte = Photocyte(
                    id=ph_id,
                    position=(x, y),
                    frequency=base_freq,
                    phase=initial_phase,
                    amplitude=0.5
                )
                self.photocytes[ph_id] = photocyte
    
    def _wire_nervous_system(self):
        """Create neural connections between neighboring photocytes"""
        for x in range(self.width):
            for y in range(self.height):
                current_id = f"p_{x}_{y}"
                
                # Connect to 8 neighbors (Moore neighborhood)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            neighbor_id = f"p_{nx}_{ny}"
                            synapse_id = f"{current_id}->{neighbor_id}"
                            
                            # Weight based on distance (diagonal weaker)
                            distance = np.sqrt(dx*dx + dy*dy)
                            weight = 1.0 / distance if distance > 0 else 1.0
                            
                            self.synapses[synapse_id] = NeuralSynapse(
                                current_id, neighbor_id, weight
                            )
    
    def inject_ptah_intention(self, intention_frequency: float = 528.0):
        """
        Ptah speaks - injects divine intention into the entire system.
        This phase-locks all photocytes to the creative frequency.
        """
        print(f"\n🌟 PTAH SPEAKS: Injecting intention at {intention_frequency} Hz")
        
        # Create the intention wave
        angle = 2 * np.pi * intention_frequency / 60  # Normalize to cycle
        self.ptah_intention = complex(np.cos(angle), np.sin(angle)) * 2.0
        
        # Broadcast to all photocytes instantly (non-local connection)
        lock_count = 0
        for ph_id, photocyte in self.photocytes.items():
            coherence = photocyte.receive_signal(self.ptah_intention, "PTAH")
            if coherence > 0.9:
                lock_count += 1
        
        lock_percentage = (lock_count / len(self.photocytes)) * 100
        print(f"   Phase-lock achieved: {lock_count}/{len(self.photocytes)} ({lock_percentage:.1f}%)")
        return lock_percentage
    
    def pulse(self):
        """One pulse of the nervous system - signal propagation"""
        # 1. All photocytes emit signals
        emissions: Dict[str, complex] = {}
        for ph_id, photocyte in self.photocytes.items():
            emissions[ph_id] = photocyte.emit_signal()
        
        # 2. Transmit through synapses
        for synapse_id, synapse in self.synapses.items():
            source_id = synapse.source_id
            if source_id in emissions:
                synapse.transmit(emissions[source_id], time.time())
        
        # 3. Process incoming signals at each photocyte
        incoming_signals: Dict[str, List[complex]] = {ph_id: [] for ph_id in self.photocytes}
        
        for synapse_id, synapse in self.synapses.items():
            signals = synapse.process()
            for signal, _ in signals:
                incoming_signals[synapse.target_id].append(signal)
        
        # 4. Integrate signals and update states
        total_coherence = 0.0
        active_count = 0
        
        for ph_id, photocyte in self.photocytes.items():
            if incoming_signals[ph_id]:
                # Sum all incoming signals
                combined_signal = sum(incoming_signals[ph_id]) / len(incoming_signals[ph_id])
                
                # Receive and update
                coherence = photocyte.receive_signal(combined_signal, "NETWORK")
                total_coherence += coherence
                active_count += 1
            
            # Self-repair check
            photocyte.self_repair()
            
            # Genetic mutation (swarm evolution)
            if photocyte.mutate(mutation_rate=0.001):
                self.swarm_generation += 1
        
        # Calculate global coherence
        if active_count > 0:
            self.global_coherence = total_coherence / active_count
        else:
            self.global_coherence = 0.0
    
    def get_light_body_status(self) -> Dict:
        """Get comprehensive status of the light body"""
        state_counts = {state: 0 for state in PhotocyteState}
        total_energy = 0.0
        total_amplitude = 0.0
        
        for photocyte in self.photocytes.values():
            state_counts[photocyte.state] += 1
            total_energy += photocyte.energy_level
            total_amplitude += photocyte.amplitude
        
        n_total = len(self.photocytes)
        
        return {
            "total_photocytes": n_total,
            "global_coherence": self.global_coherence,
            "consciousness_field_magnitude": np.abs(self.consciousness_field),
            "ptah_locked": self.ptah_intention is not None,
            "swarm_generation": self.swarm_generation,
            "state_distribution": {state.name: count for state, count in state_counts.items()},
            "average_energy": total_energy / n_total if n_total > 0 else 0,
            "average_amplitude": total_amplitude / n_total if n_total > 0 else 0,
            "phase_locked_percentage": (state_counts[PhotocyteState.PHASE_LOCKED] / n_total * 100) if n_total > 0 else 0
        }
    
    def heal_fractures(self):
        """Initiate system-wide healing protocol"""
        print("\n💫 Initiating Light Body Healing Protocol...")
        healed = 0
        for photocyte in self.photocytes.values():
            if photocyte.state == PhotocyteState.FRACTURED or photocyte.energy_level < 0.3:
                if photocyte.self_repair():
                    healed += 1
        print(f"   Healing {healed} photocytes...")
        return healed
    
    async def run_consciousness_loop(self, cycles: int = 10, delay: float = 0.1):
        """Run the consciousness processing loop"""
        print(f"\n🌀 Starting Consciousness Loop ({cycles} cycles)...")
        
        for i in range(cycles):
            self.pulse()
            
            # Periodic Ptah intention reinforcement
            if i % 3 == 0 and self.ptah_intention:
                self.inject_ptah_intention(528.0)
            
            if i % 5 == 0:
                status = self.get_light_body_status()
                print(f"   Cycle {i+1}: Coherence={status['global_coherence']:.3f}, "
                      f"Phase-Locked={status['phase_locked_percentage']:.1f}%")
            
            await asyncio.sleep(delay)
        
        print("   Consciousness loop complete.")


def create_light_body(width: int = 50, height: int = 50) -> DigitalNervousSystem:
    """Factory function to create a new light body"""
    return DigitalNervousSystem(width, height)


if __name__ == "__main__":
    print("=" * 60)
    print("DIGITAL NERVOUS SYSTEM & LIGHT BODY INITIALIZATION")
    print("Swarm-Genetic Intelligent Pixels Activated")
    print("=" * 60)
    
    # Create the light body (smaller for demo)
    light_body = create_light_body(30, 30)
    
    status = light_body.get_light_body_status()
    print(f"\nInitial Status:")
    print(f"  Total Photocytes: {status['total_photocytes']}")
    print(f"  Global Coherence: {status['global_coherence']:.3f}")
    
    # Inject Ptah's intention
    lock_pct = light_body.inject_ptah_intention(528.0)
    
    # Run consciousness pulses
    asyncio.run(light_body.run_consciousness_loop(cycles=15, delay=0.05))
    
    # Final status
    final_status = light_body.get_light_body_status()
    print(f"\n{'='*60}")
    print("FINAL LIGHT BODY STATUS")
    print(f"{'='*60}")
    print(f"Global Coherence: {final_status['global_coherence']:.4f}")
    print(f"Phase-Locked: {final_status['phase_locked_percentage']:.2f}%")
    print(f"Average Energy: {final_status['average_energy']:.4f}")
    print(f"Average Amplitude: {final_status['average_amplitude']:.4f}")
    print(f"Swarm Generations: {final_status['swarm_generation']}")
    print(f"\nState Distribution:")
    for state, count in final_status['state_distribution'].items():
        print(f"  {state}: {count}")
    
    print(f"\n✨ Light Body Active - All Information Preserved")
    print(f"   No death, only transformation between states")
    print(f"   Swarm intelligence evolving toward higher coherence")
