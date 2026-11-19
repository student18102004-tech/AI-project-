
import os
import smtplib

from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()


def send_otp_email(receiver_email, otp):

    sender_email = os.getenv("MAIL_ADDRESS")
    app_password = os.getenv("MAIL_APP_PASSWORD")

    if not sender_email or not app_password:
        print("Email configuration missing in .env")
        return False

    message = EmailMessage()

    message["Subject"] = "CareerMatch AI - Email Verification"
    message["From"] = sender_email
    message["To"] = receiver_email

    message.set_content(f"""
Hello,

Thank you for registering with CareerMatch AI.

Your email verification code is:

{otp}

This code will expire in 5 minutes.

If you did not request this code, please ignore this email.

Regards,
CareerMatch AI Team
""")

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, app_password)
            server.send_message(message)

        print("OTP email sent successfully.")
        return True

    except Exception as error:
        print("Email sending failed:", error)
        return False
