# utils/email_sender.py
import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from decouple import config as decouple_config # Using python-decouple
from dotenv import load_dotenv

load_dotenv() # Ensure .env is loaded if not already by main app for this utility

# Fallback to os.getenv if decouple can't find it (e.g., if .env not strictly used by decouple)
EMAIL_HOST = decouple_config('EMAIL_HOST', default=os.getenv('EMAIL_HOST', 'smtp.gmail.com'))
EMAIL_PORT = decouple_config('EMAIL_PORT', default=os.getenv('EMAIL_PORT', '587'), cast=int)
EMAIL_HOST_USER = decouple_config('EMAIL_HOST_USER', default=os.getenv('EMAIL_HOST_USER'))
EMAIL_HOST_PASSWORD = decouple_config('EMAIL_HOST_PASSWORD', default=os.getenv('EMAIL_HOST_PASSWORD'))
EMAIL_USE_TLS = decouple_config('EMAIL_USE_TLS', default=os.getenv('EMAIL_USE_TLS', 'True'), cast=bool)
DEFAULT_FROM_EMAIL = decouple_config('DEFAULT_FROM_EMAIL', default=os.getenv('DEFAULT_FROM_EMAIL', EMAIL_HOST_USER))


async def send_email_with_attachment(
    to_email: str,
    subject: str,
    body_html: str,
    attachment_path: str,
    attachment_filename: str
) -> tuple[bool, str]:
    if not EMAIL_HOST_USER or not EMAIL_HOST_PASSWORD:
        error_msg = "Email credentials (EMAIL_HOST_USER, EMAIL_HOST_PASSWORD) are not configured."
        print(f"Email Error: {error_msg}")
        return False, error_msg

    msg = MIMEMultipart()
    msg['From'] = DEFAULT_FROM_EMAIL
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body_html, 'html'))

    try:
        with open(attachment_path, 'rb') as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header(
            'Content-Disposition',
            f"attachment; filename= {attachment_filename}",
        )
        msg.attach(part)
    except FileNotFoundError:
        error_msg = f"Attachment file not found at {attachment_path}"
        print(f"Email Error: {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"Error attaching file: {str(e)}"
        print(f"Email Error: {error_msg}")
        return False, error_msg

    try:
        print(f"Attempting to send email to {to_email} from {DEFAULT_FROM_EMAIL} via {EMAIL_HOST}:{EMAIL_PORT}")
        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
            if EMAIL_USE_TLS:
                server.starttls()
            server.login(EMAIL_HOST_USER, EMAIL_HOST_PASSWORD)
            server.sendmail(DEFAULT_FROM_EMAIL, to_email, msg.as_string())
        print(f"Email successfully sent to {to_email}")
        return True, "Email sent successfully."
    except smtplib.SMTPAuthenticationError:
        error_msg = "SMTP Authentication Error: Check username/password (or app password for Gmail)."
        print(f"Email Error: {error_msg}")
        return False, error_msg
    except smtplib.SMTPConnectError:
        error_msg = f"SMTP Connect Error: Could not connect to {EMAIL_HOST}:{EMAIL_PORT}."
        print(f"Email Error: {error_msg}")
        return False, error_msg
    except smtplib.SMTPServerDisconnected:
        error_msg = "SMTP Server Disconnected unexpectedly."
        print(f"Email Error: {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"General error sending email: {str(e)}"
        print(f"Email Error: {error_msg}")
        return False, error_msg