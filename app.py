
import os
import random
import string
import joblib
from functools import wraps
from pathlib import Path

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash
)

from dotenv import load_dotenv

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

# Load environment variables
load_dotenv()

# Database imports
from database.db import (
    init_db,
    init_email_verification_table,
    init_profile_table,
    init_jobs_table,
    create_user,
    get_user_by_email,
    get_user_by_id,
    save_otp,
    verify_otp,
    save_profile,
    get_profile,
    delete_profile,
    count_jobs,
    get_all_jobs,
    import_jobs_from_csv
)

# Email service
from services.email_service import send_otp_email

# AI engine imports
from ai_engine import (
    bfs_career_exploration,
    predict_category,
    get_job_recommendations,
    get_skill_gaps,
    train_ml_model
)
# ============================================================
# LOAD MACHINE LEARNING MODEL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "ml_model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"

if MODEL_PATH.exists() and VECTORIZER_PATH.exists():

    ml_model = joblib.load(MODEL_PATH)
    tfidf_vectorizer = joblib.load(VECTORIZER_PATH)

    print("ML Model loaded successfully!")

else:

    print("Training ML Model...")

    ml_model, tfidf_vectorizer = train_ml_model()

    print("ML Model trained successfully!")

# ============================================================
# APP CONFIGURATION
# ============================================================

app = Flask(__name__)

app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "development-key-change-this"
)

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax"
)

EXPERIENCE_LEVELS = [
    "Beginner",
    "Intermediate",
    "Advanced"
]


# ============================================================
# LOGIN REQUIRED DECORATOR
# ============================================================

def login_required(view):

    @wraps(view)
    def wrapped(*args, **kwargs):

        if "user_id" not in session:
            flash("Please log in to continue.", "error")
            return redirect(url_for("login"))

        return view(*args, **kwargs)

    return wrapped


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


# ============================================================
# REGISTRATION
# ============================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        full_name = request.form.get(
            "full_name", ""
        ).strip()

        email = request.form.get(
            "email", ""
        ).strip().lower()

        password = request.form.get(
            "password", ""
        )

        confirm_password = request.form.get(
            "confirm_password", ""
        )

        errors = []

        if not full_name:
            errors.append("Full name is required.")

        if not email or "@" not in email:
            errors.append("Enter a valid email address.")

        if len(password) < 8:
            errors.append(
                "Password must contain at least 8 characters."
            )

        if password != confirm_password:
            errors.append("Passwords do not match.")

        if get_user_by_email(email):
            errors.append("This email is already registered.")

        if errors:

            for error in errors:
                flash(error, "error")

            return render_template("register.html"), 400

        password_hash = generate_password_hash(password)

        try:
            user_id = create_user(
                full_name,
                email,
                password_hash
            )

        except Exception as error:
            print("Registration error:", error)
            flash("Unable to create account. Please try again.", "error")
            return render_template("register.html"), 400

        code = "".join(
            random.choices(string.digits, k=6)
        )

        save_otp(user_id, code)

        email_sent = send_otp_email(email, code)

        if not email_sent:

            # Do not print or accept a debug OTP.
            session.pop("pending_user_id", None)

            flash(
                "Unable to send verification email. "
                "Please contact the administrator before retrying.",
                "error"
            )

            return redirect(url_for("register"))

        session.clear()
        session["pending_user_id"] = user_id

        flash(
            "Verification code sent to your email.",
            "success"
        )

        return redirect(url_for("verify_otp_route"))

    return render_template("register.html")


# ============================================================
# OTP VERIFICATION
# ============================================================

@app.route("/verify-otp", methods=["GET", "POST"])
def verify_otp_route():

    user_id = session.get("pending_user_id")

    if not user_id:
        return redirect(url_for("register"))

    if request.method == "POST":

        code = request.form.get("otp", "").strip()

        if not code.isdigit() or len(code) != 6:
            flash("Enter a valid 6-digit verification code.", "error")
            return render_template("verify_otp.html")

        result = verify_otp(user_id, code)

        if result == "ok":

            session.pop("pending_user_id", None)

            flash(
                "Email verified successfully. You can now log in.",
                "success"
            )

            return redirect(url_for("login"))

        flash(result, "error")

    return render_template("verify_otp.html")


# ============================================================
# LOGIN
# ============================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form.get(
            "email", ""
        ).strip().lower()

        password = request.form.get(
            "password", ""
        )

        user = get_user_by_email(email)

        if not user or not check_password_hash(
            user["password_hash"], password
        ):

            flash("Invalid email or password.", "error")
            return render_template("login.html"), 400

        if not user["is_verified"]:

            flash(
                "Please verify your email before logging in.",
                "error"
            )

            return redirect(url_for("register"))

        session.clear()

        session["user_id"] = user["id"]
        session["user_name"] = user["full_name"]

        flash("Logged in successfully.", "success")

        return redirect(url_for("dashboard"))

    return render_template("login.html")


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    flash("Logged out successfully.", "success")

    return redirect(url_for("index"))


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
@login_required
def dashboard():

    user_id = session["user_id"]

    profile = get_profile(user_id)

    job_count = count_jobs()

    return render_template(
        "dashboard.html",
        profile=profile,
        job_count=job_count,
        user_name=session.get("user_name", "User")
    )


# ============================================================
# USER PROFILE
# ============================================================

@app.route("/profile")
@login_required
def profile():

    user_id = session["user_id"]

    profile_data = get_profile(user_id)

    user = get_user_by_id(user_id)

    return render_template(
        "profile.html",
        profile=profile_data,
        user=user
    )


# ============================================================
# CREATE / UPDATE PROFILE
# ============================================================

@app.route("/profile/edit", methods=["GET", "POST"])
@login_required
def edit_profile():

    user_id = session["user_id"]

    if request.method == "POST":

        education = request.form.get(
            "education", ""
        ).strip()

        skills = request.form.get(
            "skills", ""
        ).strip()

        experience = request.form.get(
            "experience", ""
        ).strip()

        interests = request.form.get(
            "interests", ""
        ).strip()

        errors = []

        if not education:
            errors.append("Education is required.")

        if not skills:
            errors.append("Please enter your skills.")

        if experience not in EXPERIENCE_LEVELS:
            errors.append("Select a valid experience level.")

        if not interests:
            errors.append("Please enter your career interests.")

        if errors:

            for error in errors:
                flash(error, "error")

            return render_template(
                "edit_profile.html",
                profile=request.form,
                experience_levels=EXPERIENCE_LEVELS
            ), 400

        save_profile(
            user_id,
            education,
            skills,
            experience,
            interests
        )

        flash("Profile saved successfully.", "success")

        return redirect(url_for("profile"))

    existing_profile = get_profile(user_id)

    return render_template(
        "edit_profile.html",
        profile=existing_profile,
        experience_levels=EXPERIENCE_LEVELS
    )


# ============================================================
# DELETE PROFILE
# ============================================================

@app.route("/profile/delete", methods=["POST"])
@login_required
def delete_profile_route():

    user_id = session["user_id"]

    delete_profile(user_id)

    flash("Profile deleted successfully.", "success")

    return redirect(url_for("profile"))


# ============================================================
# JOB RECOMMENDATIONS
# ============================================================

@app.route("/jobs")
@login_required
def jobs():

    user_id = session["user_id"]

    profile = get_profile(user_id)

    if not profile:

        flash(
            "Please complete your profile first.",
            "error"
        )

        return redirect(url_for("profile"))

    all_jobs = get_all_jobs()

    recommendations = get_job_recommendations(
        profile["skills"],
        all_jobs
    )

    return render_template(
        "jobs.html",
        jobs=recommendations
    )


# ============================================================
# CAREER RECOMMENDATIONS
# ============================================================

@app.route("/careers")
@login_required
def careers():

    user_id = session["user_id"]

    profile = get_profile(user_id)

    if not profile:

        flash(
            "Please complete your profile first.",
            "error"
        )

        return redirect(url_for("profile"))

    predicted_category = predict_category(
    profile["skills"],
    ml_model,
    tfidf_vectorizer
)
    

    explored_categories = bfs_career_exploration(
        predicted_category
    )

    return render_template(
        "careers.html",
        predicted=predicted_category,
        explored=explored_categories
    )


# ============================================================
# SKILL GAP ANALYSIS
# ============================================================

@app.route("/skill-gaps")
@login_required
def skill_gaps():

    user_id = session["user_id"]

    profile = get_profile(user_id)

    if not profile:

        flash(
            "Please complete your profile first.",
            "error"
        )

        return redirect(url_for("profile"))

    all_jobs = get_all_jobs()

    recommendations = get_job_recommendations(
        profile["skills"],
        all_jobs
    )

    top_job = (
        recommendations[0]
        if recommendations
        else None
    )

    missing_skills = []

    if top_job:

        missing_skills = get_skill_gaps(
            profile["skills"],
            top_job["required_skills"]
        )

    return render_template(
        "skill_gaps.html",
        job=top_job,
        missing=missing_skills
    )


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def initialize_application():

    init_db()
    init_email_verification_table()
    init_profile_table()
    init_jobs_table()

    # Import jobs if the database is empty
    if count_jobs() == 0:

        jobs_csv = BASE_DIR / "datasets" / "jobs_dataset.csv"

        if jobs_csv.exists():
            import_jobs_from_csv(jobs_csv)
            print("Jobs dataset imported successfully!")
        else:
            print("Warning: Jobs dataset CSV file not found.")

    print("Total jobs in database:", count_jobs())

# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    initialize_application()

    app.run(debug=True)
