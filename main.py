import os
import json
import hashlib
from fastapi import FastAPI, HTTPException
from qiskit import QuantumCircuit, Aer, execute
import redis
from pydantic import BaseModel

# Initialize FastAPI
app = FastAPI()

# Initialize Redis Client for Context Storage
redis_client = redis.StrictRedis(host='localhost', port=6379, decode_responses=True)

# Quantum Processor Class
class QuantumProcessor:
    def __init__(self):
        self.backend = Aer.get_backend('statevector_simulator')

    def process_text(self, text: str):
        # Quantum Circuit for Text Processing
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        # Execute Circuit
        job = execute(qc, self.backend)
        result = job.result()
        statevector = result.get_statevector()

        # Hash the text input for storage
        context_hash = hashlib.sha256(text.encode()).hexdigest()
        return context_hash, statevector.tolist()

# Pydantic Model for Input Validation
class InputData(BaseModel):
    text: str

# Route: Process Input
@app.post("/process/")
async def process_input(data: InputData):
    processor = QuantumProcessor()
    context_hash, statevector = processor.process_text(data.text)

    # Store the statevector in Redis
    redis_client.set(context_hash, json.dumps({"statevector": statevector}))

    return {"context_hash": context_hash, "statevector": statevector}

# Route: Retrieve Statevector
@app.get("/retrieve/{context_hash}")
async def retrieve_state(context_hash: str):
    state = redis_client.get(context_hash)
    if not state:
        raise HTTPException(status_code=404, detail="Context not found or expired")
    return json.loads(state)
