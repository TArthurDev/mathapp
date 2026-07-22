import json
import os
import time
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from huggingface_hub import AsyncInferenceClient
 
load_dotenv()
 
app = FastAPI(title="Math Competition Tracker API")
 
# NOTE: allow_credentials must be False when allow_origins is "*" (browsers reject the
# combination of wildcard origin + credentials). This app doesn't use cookies for auth
# (student_code is passed explicitly in the request body/path), so this is safe.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# The Hugging Face model used to grade submissions. Override with the HF_MODEL env var
# if your club wants to use a different instruct model.
HF_MODEL = os.environ.get("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
 
DATA_DIR = "data"
PROBLEMS_FILE = os.path.join(DATA_DIR, "seed_problems.json")
STUDENTS_FILE = os.path.join(DATA_DIR, "students.json")
HISTORY_FILE = os.path.join(DATA_DIR, "history.json")
USERS_FILE = os.path.join(DATA_DIR, "users.json")
 
ACCESS_CODES = {
    "lukep": "operator",
    "sarahm": "student",
    "alexw": "student"
}
 
os.makedirs(DATA_DIR, exist_ok=True)
 
all_club_submissions: List[Dict[str, Any]] = []
 
 
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
    rubric: str = ""
    image_url: Optional[str] = ""
    lesson_link: Optional[str] = ""
    url: Optional[str] = ""
 
 
class ProblemCreate(BaseModel):
    id: str
    title: str
    concept: str
    question: str
    image_url: Optional[str] = ""
    reference_solution: str
    lesson_link: Optional[str] = ""
    url: Optional[str] = ""
 
 
class ProblemUpdate(BaseModel):
    title: Optional[str] = None
    concept: Optional[str] = None
    question: Optional[str] = None
    official_answer: Optional[str] = None
    rubric: Optional[str] = None
    image_url: Optional[str] = None
    lesson_link: Optional[str] = None
    url: Optional[str] = None
 
 
class SubmitRequest(BaseModel):
    problem_id: str
    student_code: str
    exact_answer: str = ""
    explanation: str = ""
 
 
class SubmitResponse(BaseModel):
    correctness_score: float
    explanation_score: float
    rigor_score: float
    justification: str
    teacher_solution: str = ""
    tutor_hint: str
    resources_used: str = ""
    streak: int
    url: Optional[str] = ""
 
 
class PerformanceResponse(BaseModel):
    averages: Dict[str, float]
    total_submissions: int
 
 
class StudentOverview(BaseModel):
    student_code: str
    performance_metrics: Dict[str, float]
    metric_breakdown: Dict[str, float]
    total_submissions: int
    submissions: List[Dict[str, Any]]
 
 
def load_json(file_path: str, default: Any) -> Any:
    if not os.path.exists(file_path):
        return default
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception:
        return default
 
 
def save_json(file_path: str, data: Any) -> None:
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
 
 
def sanitize_problems(problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(problems, list):
        return []
    cleaned: List[Dict[str, Any]] = []
    for index, problem in enumerate(problems):
        if not isinstance(problem, dict):
            continue
        question = str(problem.get("question", "")).lower()
        rubric = str(problem.get("rubric", "")).lower()
        bad_problem = (
            not str(problem.get("id", "")).strip()
            or "1+1" in question
            or "1 + 1" in question
            or "1+1" in rubric
            or "1 + 1" in rubric
            or "incorrect answer is 3" in rubric
        )
        if bad_problem:
            continue
        normalized = dict(problem)
        normalized.setdefault("id", f"problem_{index + 1}")
        normalized.setdefault("title", "Untitled Problem")
        normalized.setdefault("concept", "General")
        normalized.setdefault("question", "")
        normalized.setdefault("official_answer", "")
        normalized.setdefault("rubric", "")
        normalized.setdefault("image_url", "")
        normalized.setdefault("lesson_link", "")
        normalized.setdefault("url", "")
        cleaned.append(normalized)
    return cleaned
 
 
if not os.path.exists(PROBLEMS_FILE):
    sample_problems = [
        {
            "id": "prob1", "title": "Prime Time", "concept": "Number Theory",
            "question": "How many positive integers less than 100 have exactly three positive divisors?",
            "official_answer": "4",
            "rubric": "Squares of primes. Primes <10: 2,3,5,7 => 4,9,25,49. Ans: 4.",
            "image_url": "", "lesson_link": "", "url": ""
        },
        {
            "id": "prob2", "title": "Arranging Letters", "concept": "Combinatorics",
            "question": "In how many ways can the letters of the word 'MATH' be arranged if the vowels must be together?",
            "official_answer": "24",
            "rubric": "There is only 1 vowel ('A'). It is always together with itself, so all 4! = 24 arrangements count.",
            "image_url": "", "lesson_link": "", "url": ""
        }
    ]
    save_json(PROBLEMS_FILE, sample_problems)
else:
    raw_problems = load_json(PROBLEMS_FILE, [])
    cleaned_problems = sanitize_problems(raw_problems)
    if cleaned_problems != raw_problems:
        save_json(PROBLEMS_FILE, cleaned_problems)
 
if not os.path.exists(STUDENTS_FILE):
    save_json(STUDENTS_FILE, list(ACCESS_CODES.keys()))
 
if not os.path.exists(HISTORY_FILE):
    save_json(HISTORY_FILE, [])
 
if not os.path.exists(USERS_FILE):
    save_json(USERS_FILE, {})
 
# Load any submissions already on disk into memory so /api/admin endpoints and the
# operator dashboard reflect history across server restarts.
all_club_submissions = list(load_json(HISTORY_FILE, []))
 
 
@app.get("/", response_class=HTMLResponse)
def read_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(current_dir, "index.html"),
        os.path.join(current_dir, "frontend", "math.html"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    return "<h1>Backend Running</h1>"
 
 
@app.post("/api/login", response_model=LoginResponse)
def login(request: LoginRequest):
    student_code = request.student_code.lower().strip()
    if student_code in ACCESS_CODES:
        return LoginResponse(success=True, message="Login successful", role=ACCESS_CODES[student_code])
    return LoginResponse(success=False, message="Invalid access code")
 
 
@app.get("/api/problems", response_model=List[Problem])
def get_problems():
    problems = load_json(PROBLEMS_FILE, [])
    return sanitize_problems(problems)
 
 
@app.post("/api/problems", response_model=Problem)
def add_problem(problem: ProblemCreate):
    problems = sanitize_problems(load_json(PROBLEMS_FILE, []))
    for p in problems:
        if p.get("id") == problem.id:
            raise HTTPException(status_code=400, detail="Problem ID already exists")
    new_problem = {
        "id": problem.id, "title": problem.title, "concept": problem.concept,
        "question": problem.question, "official_answer": "",
        "rubric": problem.reference_solution, "image_url": problem.image_url or "",
        "lesson_link": problem.lesson_link or "",
        "url": problem.url or ""
    }
    problems.append(new_problem)
    save_json(PROBLEMS_FILE, problems)
    return new_problem
 
 
@app.delete("/api/problems/{problem_id}")
def delete_problem(problem_id: str):
    problems = sanitize_problems(load_json(PROBLEMS_FILE, []))
    updated_problems = [p for p in problems if str(p.get("id", "")) != str(problem_id)]
    if len(problems) == len(updated_problems):
        raise HTTPException(status_code=404, detail="Problem not found")
    save_json(PROBLEMS_FILE, updated_problems)
    return {"success": True, "message": f"Problem {problem_id} deleted successfully."}
 
 
@app.put("/api/problems/{problem_id}", response_model=Problem)
def update_problem(problem_id: str, updates: ProblemUpdate):
    problems = sanitize_problems(load_json(PROBLEMS_FILE, []))
    for problem in problems:
        if str(problem.get("id", "")) == str(problem_id):
            if updates.title is not None:
                problem["title"] = updates.title
            if updates.concept is not None:
                problem["concept"] = updates.concept
            if updates.question is not None:
                problem["question"] = updates.question
            if updates.official_answer is not None:
                problem["official_answer"] = updates.official_answer
            if updates.rubric is not None:
                problem["rubric"] = updates.rubric
            if updates.image_url is not None:
                problem["image_url"] = updates.image_url
            if updates.lesson_link is not None:
                problem["lesson_link"] = updates.lesson_link
            if updates.url is not None:
                problem["url"] = updates.url
            save_json(PROBLEMS_FILE, problems)
            return problem
    raise HTTPException(status_code=404, detail="Problem not found")
 
 
def _extract_json_object(text: str) -> Dict[str, Any]:
    """Pull the first {...} JSON object out of a model response and parse it."""
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        raise ValueError("No JSON object found in model output")
    return json.loads(text[start_idx:end_idx + 1])
 
 
@app.post("/api/submit", response_model=SubmitResponse)
async def submit_solution(request: SubmitRequest):
    problems = load_json(PROBLEMS_FILE, [])
    problem = next((p for p in problems if p["id"] == request.problem_id), None)
    if problem is None:
        raise HTTPException(status_code=404, detail="Problem not found")
 
    question = problem.get("question", "")
    rubric = problem.get("rubric", "") or problem.get("official_answer", "")
    lesson_link = problem.get("lesson_link", "")
    problem_url = problem.get("url", "")
 
    fallback = {
        "correctness_score": 0.0,
        "explanation_score": 0.0,
        "rigor_score": 0.0,
        "justification": "We couldn't reach the grading service right now, so this submission wasn't scored.",
        "teacher_solution": "",
        "tutor_hint": "Please try submitting again in a moment.",
        "resources_used": "",
        "url": problem_url
    }
 
    feedback = fallback
    try:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if not hf_token:
            raise RuntimeError(
                "No Hugging Face API token configured. Set the HF_TOKEN environment variable."
            )
 
        client = AsyncInferenceClient(api_key=hf_token)
 
        system_prompt = (
            "You are a supportive, professional math teacher speaking directly to the student. "
            "Always use second-person language such as 'You', 'Your solution', or 'Your work'. "
            "Never say 'The student'. "
            "You will receive TWO separate inputs from the student: (1) Student's Final Answer and (2) Student's Explanation. "
            "Score THREE criteria on a scale from 0 to 10: correctness_score, explanation_score, and rigor_score. "
            "STRICT BOUNDARY RULE: Evaluate the two inputs independently. "
            "- correctness_score MUST depend ONLY on whether Student's Final Answer matches the expected correct value implied by the provided reference solution / grading criteria. "
            "- If the final answer is wrong, explanation_score and rigor_score must NOT be reduced merely because the final number was incorrect. "
            "- explanation_score MUST depend ONLY on the quality of the student's reasoning/step-by-step explanation for the math task. "
            "- rigor_score MUST depend ONLY on the clarity, completeness, and mathematical justification of the reasoning steps. "
            "CRITICAL RULE: You must accept the provided Rubric and Reference Solution as absolute, objective truth for this universe, even if it contradicts the laws of mathematics, logic, or reality in the real world. If the rubric states an incorrect mathematical fact and the student follows that fact, you must award full points according to the rubric guidelines. Do not apply real-world corrections. "
            "ANTI-AI SLOP RULE: You must never use the words 'rubric', 'reference solution', 'universe', or 'system prompt' in your feedback or justification fields. Act completely as a human teacher. Instead of saying 'according to the rubric', say 'according to standard grading criteria', 'based on the expected steps', or simply explain the mathematical expectations naturally. "
            "In your feedback, write a clear, concise teacher explanation showing exactly how the student should conceptually approach solving the problem step-by-step. "
            "Also provide a tutor hint that gives a conceptual hint without revealing the direct answer. "
            "If the student scores low, explicitly mention the provided lesson resource link in the feedback text so the student knows where to study next. "
            "Return ONLY a JSON object wrapped in a markdown code block like:\n"
            "```json\n{\n  \"correctness_score\": 0.0,\n  \"explanation_score\": 0.0,\n  \"rigor_score\": 0.0,\n  \"justification\": \"A clear, concise teacher-style explanation of the student's work and the correct conceptual approach\",\n  \"teacher_solution\": \"A short step-by-step explanation of how to solve the problem conceptually\",\n  \"tutor_hint\": \"A conceptual hint without revealing the answer\",\n  \"resources_used\": \"A short note naming the lesson resource if the student should review it\"\n}\n```\n"
        )
 
        user_content = (
            f"Problem Statement:\n{question}\n\n"
            f"Reference Solution / Rubric:\n{rubric}\n\n"
            f"Lesson Resource:\n{lesson_link if lesson_link else 'No lesson link provided.'}\n\n"
            f"Practice URL:\n{problem_url if problem_url else 'No practice URL provided.'}\n\n"
            f"Student's Final Answer:\n{request.exact_answer}\n\n"
            f"Student's Explanation:\n{request.explanation}"
        )
 
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
 
        chat_completion = await client.chat.completions.create(
            model=HF_MODEL,
            messages=messages,
            max_tokens=512,
            temperature=0.0,
        )
        generated_text = chat_completion.choices[0].message.content
        parsed = _extract_json_object(generated_text)
 
        correctness = max(0.0, min(10.0, float(parsed.get("correctness_score", 0.0))))
        explanation = max(0.0, min(10.0, float(parsed.get("explanation_score", 0.0))))
        rigor = max(0.0, min(10.0, float(parsed.get("rigor_score", 0.0))))
 
        feedback = {
            "correctness_score": correctness,
            "explanation_score": explanation,
            "rigor_score": rigor,
            "justification": str(parsed.get("justification", "")),
            "teacher_solution": str(parsed.get("teacher_solution", "")),
            "tutor_hint": str(parsed.get("tutor_hint", "Keep at it!")),
            "resources_used": str(parsed.get("resources_used", "")),
            "url": problem_url
        }
    except Exception as e:
        print(f"Grading error for problem {request.problem_id}: {e}")
        feedback = fallback
 
    user_profiles = load_json(USERS_FILE, {})
    if not isinstance(user_profiles, dict):
        user_profiles = {}
 
    student_key = request.student_code.lower().strip()
    profile = user_profiles.get(student_key, {"streak": 0})
    if not isinstance(profile, dict):
        profile = {"streak": 0}
 
    current_streak = int(profile.get("streak", 0))
    if feedback["correctness_score"] >= 7.0:
        current_streak += 1
    else:
        current_streak = 0
 
    profile["streak"] = current_streak
    user_profiles[student_key] = profile
    save_json(USERS_FILE, user_profiles)
    feedback["streak"] = current_streak
 
    history = load_json(HISTORY_FILE, [])
    submission_entry = {
        "student_code": student_key,
        "problem_id": request.problem_id,
        "exact_answer": request.exact_answer,
        "explanation": request.explanation,
        "timestamp": time.time(),
        **feedback
    }
    history.append(submission_entry)
    save_json(HISTORY_FILE, history)
    all_club_submissions.append(submission_entry)
 
    return feedback
 
 
@app.get("/api/student/{student_code}/performance", response_model=PerformanceResponse)
def get_performance(student_code: str):
    student_code = student_code.lower().strip()
    history = load_json(HISTORY_FILE, [])
    student_history = [e for e in history if e.get("student_code") == student_code]
    if not student_history:
        return PerformanceResponse(averages={}, total_submissions=0)
    problems = load_json(PROBLEMS_FILE, [])
    problem_concept = {p["id"]: p["concept"] for p in problems}
    concept_scores: Dict[str, List[float]] = {}
    for entry in student_history:
        pid = entry.get("problem_id")
        concept = problem_concept.get(pid)
        if not concept:
            continue
        avg_score = (entry.get("correctness_score", 0) + entry.get("explanation_score", 0) + entry.get("rigor_score", 0)) / 3.0
        concept_scores.setdefault(concept, []).append(avg_score)
    averages = {c: round(sum(s) / len(s), 2) for c, s in concept_scores.items()}
    return PerformanceResponse(averages=averages, total_submissions=len(student_history))
 
 
@app.get("/api/admin/audit")
def get_admin_audit():
    return all_club_submissions
 
 
@app.get("/api/admin/submissions/{student_id}")
def get_student_submissions(student_id: str):
    sid = student_id.lower().strip()
    return [s for s in all_club_submissions if s.get("student_code") == sid]
 
 
@app.post("/api/admin/reset")
def reset_student_data():
    global all_club_submissions
    all_club_submissions = []
    save_json(HISTORY_FILE, [])
    user_profiles = load_json(USERS_FILE, {})
    if isinstance(user_profiles, dict):
        for student_code in user_profiles:
            if isinstance(user_profiles[student_code], dict):
                user_profiles[student_code]["streak"] = 0
        save_json(USERS_FILE, user_profiles)
    return {"status": "success", "message": "All testing data cleared"}
 
 
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
        concept_scores: Dict[str, List[float]] = {}
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
        averages = {c: round(sum(s) / len(s), 2) for c, s in concept_scores.items()}
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
