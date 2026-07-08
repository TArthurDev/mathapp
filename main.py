import json
import os
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from typing import List, Dict, Any, Optional
from huggingface_hub import AsyncInferenceClient

app = FastAPI(title="Math Competition Tracker API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "data"
PROBLEMS_FILE = os.path.join(DATA_DIR, "seed_problems.json")
STUDENTS_FILE = os.path.join(DATA_DIR, "students.json")
HISTORY_FILE = os.path.join(DATA_DIR, "history.json")

ACCESS_CODES = {
    "lukep": "operator",
    "sarahm": "student",
    "alexw": "student"
}

os.makedirs(DATA_DIR, exist_ok=True)

class LoginRequest(BaseModel):
    student_code: str

class LoginResponse(BaseModel):
    success: bool
    message: str = ""
    role: str = ""

class Problem(BaseModel):
    id: str
    title: str
    concept: str
    question: str
    official_answer: str = ""
    rubric: str
    image_url: Optional[str] = ""

class ProblemCreate(BaseModel):
    id: str
    title: str
    concept: str
    question: str
    image_url: Optional[str] = ""
    reference_solution: str

class SubmitRequest(BaseModel):
    problem_id: str
    student_code: str
    text_work: str

class SubmitResponse(BaseModel):
    correctness_score: float
    explanation_score: float
    rigor_score: float
    justification: str
    tutor_hint: str

class PerformanceResponse(BaseModel):
    averages: Dict[str, float]
    total_submissions: int

# Expanded schema to feed our new multi-layered dashboard
class StudentOverview(BaseModel):
    student_code: str
    performance_metrics: Dict[str, float]
    metric_breakdown: Dict[str, float]  # correctness, explanation, rigor/justification
    total_submissions: int
    submissions: List[Dict[str, Any]] # Raw submissions for drill-down views

def load_json(file_path: str, default: Any) -> Any:
    if not os.path.exists(file_path):
        return default
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except:
        return default

def save_json(file_path: str, data: Any) -> None:
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

if not os.path.exists(PROBLEMS_FILE):
    sample_problems = [
        {"id": "prob1", "title": "Prime Time", "concept": "Number Theory", "question": "How many positive integers less than 100 have exactly three positive divisors?", "official_answer": "4", "rubric": "Squares of primes. Primes <10: 2,3,5,7 => 4,9,25,49. Ans: 4.", "image_url": ""},
        {"id": "prob2", "title": "Arranging Letters", "concept": "Combinatorics", "question": "In how many ways can the letters of the word 'MATH' be arranged if the vowels must be together?", "official_answer": "24", "rubric": "There is only 1 vowel ('A'). It is always together with itself.", "image_url": ""}
    ]
    save_json(PROBLEMS_FILE, sample_problems)

if not os.path.exists(STUDENTS_FILE):
    save_json(STUDENTS_FILE, list(ACCESS_CODES.keys()))

if not os.path.exists(HISTORY_FILE):
    save_json(HISTORY_FILE, [])

@app.get("/", response_class=HTMLResponse)
def read_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [os.path.join(current_dir, "frontend", "index.html"), os.path.join(current_dir, "frontend", "index")]
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    return f"<h1>Backend Running</h1>"

@app.post("/api/login", response_model=LoginResponse)
def login(request: LoginRequest):
    student_code = request.student_code.lower().strip()
    if student_code in ACCESS_CODES:
        return LoginResponse(success=True, message="Login successful", role=ACCESS_CODES[student_code])
    return LoginResponse(success=False, message="Invalid access code")

@app.get("/api/problems", response_model=List[Problem])
def get_problems():
    return load_json(PROBLEMS_FILE, [])

@app.post("/api/problems", response_model=Problem)
def add_problem(problem: ProblemCreate):
    problems = load_json(PROBLEMS_FILE, [])
    for p in problems:
        if p["id"] == problem.id:
            raise HTTPException(status_code=400, detail="Problem ID already exists")
    new_problem = {
        "id": problem.id, "title": problem.title, "concept": problem.concept,
        "question": problem.question, "official_answer": "", 
        "rubric": problem.reference_solution, "image_url": problem.image_url
    }
    problems.append(new_problem)
    save_json(PROBLEMS_FILE, problems)
    return new_problem

# NEW ENDPOINT: Delete a problem
@app.delete("/api/problems/{problem_id}")
def delete_problem(problem_id: str):
    problems = load_json(PROBLEMS_FILE, [])
    updated_problems = [p for p in problems if p["id"] != problem_id]
    
    if len(problems) == len(updated_problems):
        raise HTTPException(status_code=404, detail="Problem not found")
        
    save_json(PROBLEMS_FILE, updated_problems)
    return {"success": True, "message": f"Problem {problem_id} deleted successfully."}

@app.post("/api/submit", response_model=SubmitResponse)
async def submit_solution(request: SubmitRequest):
    # Load problem details for context
    problems = load_json(PROBLEMS_FILE, [])
    problem = next((p for p in problems if p["id"] == request.problem_id), None)
    question = problem.get("question", "") if problem else ""
    rubric = problem.get("rubric", "") if problem else ""
    # If rubric empty, fallback to official_answer
    if not rubric and problem:
        rubric = problem.get("official_answer", "")

    # Default fallback response
    fallback = {
        "correctness_score": 0.0,
        "explanation_score": 0.0,
        "rigor_score": 0.0,
        "justification": "API error: unable to evaluate submission.",
        "tutor_hint": "Please try again later."
    }

    try:
        # Get Hugging Face API token from environment, with hardcoded fallback
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "hf_fallback_token_for_demo"
        # Create Hugging Face Async Inference client
        client = AsyncInferenceClient(
            api_key="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"
        )

        # Model to use - can be configured via environment variable, default to math-optimized model
        hf_model = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")

        # System prompt instructing to output JSON in a markdown code block
        system_prompt_json = (
            "You are a strict math competition judge. Given a student's solution text, the problem statement, "
            "and the reference solution/rubric, score the student's work on three criteria: correctness, explanation, "
            "and rigor/justification, each on a scale from 0 to 10. Provide a justification for the scores, pointing out "
            "any missing steps, errors, or missing explanations. Also provide a tutor hint that gives a conceptual hint "
            "without revealing the direct answer. Return ONLY a JSON object wrapped in a markdown code block like:\n"
            "```json\n{\n  \"correctness_score\": 0.0,\n  \"explanation_score\": 0.0,\n  \"rigor_score\": 0.0,\n  \"justification\": \"string explaining the deductions or points earned\",\n  \"tutor_hint\": \"string providing a conceptual hint without revealing the direct answer\"\n}\n```\n"
        )

        # System and user prompts for chat completion
        system_prompt = system_prompt_json
        user_content = f"""Problem Statement:
{question}

Reference Solution / Rubric:
{rubric}

Student Solution:
{request.text_work}"""

        # Generate text using the chat completion API asynchronously
        # Parameters for deterministic output similar to temperature=0
        chat_completion = await client.chat.completions.create(
            model=hf_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            max_tokens=512,
            temperature=0.0,
        )
        generated_text = chat_completion.choices[0].message.content
        print(f"RAW MODEL OUTPUT: {generated_text}")

        # Find the absolute first '{' and absolute last '}' to strip all conversational text
        start_idx = generated_text.find("{")
        end_idx = generated_text.rfind("}")

        if start_idx != -1 and end_idx != -1:
            result_text = generated_text[start_idx:end_idx + 1]
        else:
            result_text = generated_text

        # Parse JSON
        import json as pyjson
        parsed = pyjson.loads(result_text)
        # Ensure all required keys present and are correct types
        correctness = float(parsed.get("correctness_score", 0.0))
        explanation = float(parsed.get("explanation_score", 0.0))
        rigor = float(parsed.get("rigor_score", 0.0))
        justification = str(parsed.get("justification", ""))
        tutor_hint = str(parsed.get("tutor_hint", ""))
        # Clamp scores to 0-10
        correctness = max(0.0, min(10.0, correctness))
        explanation = max(0.0, min(10.0, explanation))
        rigor = max(0.0, min(10.0, rigor))
        feedback = {
            "correctness_score": correctness,
            "explanation_score": explanation,
            "rigor_score": rigor,
            "justification": justification,
            "tutor_hint": tutor_hint
        }
    except Exception as e:
        # Log error if desired (print for simplicity)
        print(f"Hugging Face API error: {e}")
        feedback = fallback

    # Save submission to history
    history = load_json(HISTORY_FILE, [])
    submission_entry = {
        "student_code": request.student_code.lower().strip(),
        "problem_id": request.problem_id,
        "text_work": request.text_work,
        "timestamp": time.time(),
        **feedback
    }
    history.append(submission_entry)
    save_json(HISTORY_FILE, history)
    return feedback

@app.get("/api/student/{student_code}/performance", response_model=PerformanceResponse)
def get_performance(student_code: str):
    history = load_json(HISTORY_FILE, [])
    student_history = [e for e in history if e.get("student_code") == student_code]
    if not student_history:
        return PerformanceResponse(averages={}, total_submissions=0)

    problems = load_json(PROBLEMS_FILE, [])
    problem_concept = {p["id"]: p["concept"] for p in problems}

    concept_scores = {}
    for entry in student_history:
        pid = entry.get("problem_id")
        concept = problem_concept.get(pid)
        if not concept: continue
        avg_score = (entry.get("correctness_score", 0) + entry.get("explanation_score", 0) + entry.get("rigor_score", 0)) / 3.0
        concept_scores.setdefault(concept, []).append(avg_score)

    averages = {c: round(sum(s)/len(s), 2) for c, s in concept_scores.items()}
    return PerformanceResponse(averages=averages, total_submissions=len(student_history))

@app.get("/api/operator/students", response_model=List[StudentOverview])
def get_operator_student_overview():
    students = load_json(STUDENTS_FILE, [])
    history = load_json(HISTORY_FILE, [])
    problems = load_json(PROBLEMS_FILE, [])
    problem_concept = {p["id"]: p["concept"] for p in problems}

    overview_list = []
    for student_code in students:
        student_history = [e for e in history if e.get("student_code") == student_code]
        
        if not student_history:
            overview_list.append(StudentOverview(
                student_code=student_code, performance_metrics={},
                metric_breakdown={"correctness": 0, "explanation": 0, "justification": 0},
                total_submissions=0, submissions=[]
            ))
            continue

        concept_scores = {}
        c_tot, e_tot, j_tot = 0.0, 0.0, 0.0
        
        for entry in student_history:
            pid = entry.get("problem_id")
            concept = problem_concept.get(pid, "Unknown")
            
            c = entry.get("correctness_score", 0)
            e = entry.get("explanation_score", 0)
            j = entry.get("rigor_score", 0)
            
            c_tot += c
            e_tot += e
            j_tot += j
            
            avg_score = (c + e + j) / 3.0
            concept_scores.setdefault(concept, []).append(avg_score)

        n = len(student_history)
        averages = {c: round(sum(s)/len(s), 2) for c, s in concept_scores.items()}
        breakdown = {
            "correctness": round(c_tot / n, 2),
            "explanation": round(e_tot / n, 2),
            "justification": round(j_tot / n, 2)
        }

        overview_list.append(StudentOverview(
            student_code=student_code,
            performance_metrics=averages,
            metric_breakdown=breakdown,
            total_submissions=n,
            submissions=student_history
        ))

    return overview_list