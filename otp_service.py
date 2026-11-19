
import os
import hmac
import hashlib
import secrets

from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

from database.db import get_connection
from services.email_service import send_otp_email

load_dotenv()


def create_otp_hash(email, otp):

    secret_key = os.getenv("FLASK_SECRET_KEY")

    if not secret_key:
        raise RuntimeError("FLASK_SECRET_KEY is missing")

    message = f"{email.lower()}:{otp}"

    return hmac.new(
        secret_key.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()


def send_verification_code(email):

    email = email.strip().lower()

    connection = get_connection()
    cursor = connection.cursor()

    current_time = datetime.now(timezone.utc)

    # Check resend cooldown
    cursor.execute(
        "SELECT last_sent_at FROM email_verifications WHERE email = ?",
        (email,)
    )

    existing = cursor.fetchone()

    if existing:
        last_sent = datetime.fromisoformat(existing["last_sent_at"])

        if current_time - last_sent < timedelta(seconds=60):
            connection.close()
            return False, "Please wait 60 seconds before requesting another OTP."

    # Generate random 6-digit OTP
    otp = f"{secrets.randbelow(900000) + 100000}"

    otp_hash = create_otp_hash(email, otp)

    expiry_time = current_time + timedelta(minutes=5)

    # Send actual email
    email_sent = send_otp_email(email, otp)

    if not email_sent:
        connection.close()
        return False, "Unable to send email. Please try again."

    # Save hashed OTP in database
    cursor.execute("""
        INSERT INTO email_verifications
        (email, otp_hash, expires_at, attempts, last_sent_at, verified)
        VALUES (?, ?, ?, 0, ?, 0)

        ON CONFLICT(email) DO UPDATE SET
            otp_hash = excluded.otp_hash,
            expires_at = excluded.expires_at,
            attempts = 0,
            last_sent_at = excluded.last_sent_at,
            verified = 0
    """, (
        email,
        otp_hash,
        expiry_time.isoformat(),
        current_time.isoformat()
    ))

    connection.commit()
    connection.close()

    return True, "Verification code sent successfully!"


def verify_otp(email, entered_otp):

    email = email.strip().lower()
    entered_otp = entered_otp.strip()

    if not entered_otp.isdigit() or len(entered_otp) != 6:
        return False, "Enter a valid 6-digit OTP."

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        "SELECT * FROM email_verifications WHERE email = ?",
        (email,)
    )

    record = cursor.fetchone()

    if not record:
        connection.close()
        return False, "No OTP found. Request a new code."

    if record["verified"]:
        connection.close()
        return True, "Email already verified."

    if record["attempts"] >= 5:
        connection.close()
        return False, "Maximum attempts reached. Request a new OTP."

    current_time = datetime.now(timezone.utc)
    expiry_time = datetime.fromisoformat(record["expires_at"])

    if current_time >= expiry_time:
        connection.close()
        return False, "OTP expired. Request a new code."

    expected_hash = record["otp_hash"]
    entered_hash = create_otp_hash(email, entered_otp)

    if not hmac.compare_digest(expected_hash, entered_hash):

        cursor.execute("""
            UPDATE email_verifications
            SET attempts = attempts + 1
            WHERE email = ?
        """, (email,))

        connection.commit()
        connection.close()

        return False, "Incorrect OTP. Please try again."

    cursor.execute("""
        UPDATE email_verifications
        SET verified = 1
        WHERE email = ?
    """, (email,))

    connection.commit()
    connection.close()

    return True, "Email verified successfully!"
