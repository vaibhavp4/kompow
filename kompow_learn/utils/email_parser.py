# This file will contain utility functions for parsing emails.
import imaplib
import email
from email.header import decode_header
from email.utils import parseaddr
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import re
import io

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None
    # print("pypdf not found. PDF attachment parsing will be skipped. Please install it.") # Less verbose

try:
    import docx
except ImportError:
    docx = None
    # print("python-docx not found. DOCX attachment parsing will be skipped. Please install it.") # Less verbose

# Assuming web_crawler.py is in the same directory (utils)
from .web_crawler import fetch_url_content
# Assuming knowledge_base.py is in the same directory (utils)
from .knowledge_base import get_user_knowledge_base, add_document_to_kb, sanitize_table_name
# query_knowledge_base is not directly used in this file after this change.


def connect_to_mailbox(host, user, password):
    try:
        mail = imaplib.IMAP4_SSL(host)
        mail.login(user, password)
        # print("Successfully connected to the mailbox.") # Less verbose in loop
        return mail
    except imaplib.IMAP4.error as e:
        print(f"Error connecting to mailbox: {e}")
        return None

def extract_urls_from_text(text):
    if not text: return []
    url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    urls = re.findall(url_pattern, text)
    normalized_urls = []
    for u in urls:
        if u.startswith('www.') and not u.startswith(('http://', 'https://')):
            normalized_urls.append('http://' + u)
        else:
            normalized_urls.append(u)
    return list(set(normalized_urls))

def decode_filename(filename):
    if filename is None: return None
    decoded_filename_parts = []
    for part, charset in decode_header(filename):
        if isinstance(part, bytes):
            decoded_filename_parts.append(part.decode(charset or 'utf-8', errors='replace'))
        else:
            decoded_filename_parts.append(part)
    return "".join(decoded_filename_parts)

def extract_attachments(message_parts, email_subject_for_metadata: str, email_id_for_doc_id: str):
    attachments = []
    for part in message_parts:
        if part.get_content_disposition() == 'attachment':
            filename = decode_filename(part.get_filename())
            if filename:
                content_type = part.get_content_type()
                payload = part.get_payload(decode=True)
                attachment_data_text = None
                file_ext = filename.lower().split('.')[-1]

                if file_ext == "txt":
                    try: attachment_data_text = payload.decode('utf-8', errors='replace')
                    except Exception as e: attachment_data_text = f"Error decoding .txt: {e}"
                elif file_ext == "pdf":
                    if PdfReader:
                        try:
                            reader = PdfReader(io.BytesIO(payload))
                            attachment_data_text = "".join([p.extract_text() or "" for p in reader.pages]).strip() or "No text in PDF."
                        except Exception as e: attachment_data_text = f"Error parsing PDF: {e}"
                    else: attachment_data_text = f"PDF: {filename} (pypdf not installed)"
                elif file_ext == "docx":
                    if docx:
                        try:
                            attachment_data_text = "\n".join([p.text for p in docx.Document(io.BytesIO(payload)).paragraphs]).strip() or "No text in DOCX."
                        except Exception as e: attachment_data_text = f"Error parsing DOCX: {e}"
                    else: attachment_data_text = f"DOCX: {filename} (python-docx not installed)"
                else:
                    attachment_data_text = f"Attachment type '{file_ext}' not parsed: {filename}"

                attachments.append({
                    "filename": filename, "content_type": content_type,
                    "data": attachment_data_text,
                    "doc_id_base": f"email_{email_id_for_doc_id}_attachment_{sanitize_table_name(filename)}"
                })
    return attachments

def parse_email_data(mail_server, max_emails_to_process: int = 10): # Added limit
    if mail_server is None: return []
    status, _ = mail_server.select("INBOX") # Or "UNSEEN"
    if status != "OK": print("Error selecting INBOX"); return []

    # Search for UNSEEN emails. If none, can search for ALL for testing.
    # For production, stick to UNSEEN or a date-based search.
    status, email_ids_bytes = mail_server.search(None, "UNSEEN")
    if status != "OK" or not email_ids_bytes[0].strip():
        # print("No unseen emails found. Trying ALL emails for this session (for testing/dev).") # Optional: for dev
        # status, email_ids_bytes = mail_server.search(None, "ALL")
        # if status != "OK" or not email_ids_bytes[0].strip():
        print("No emails found matching search criteria (UNSEEN).")
        return []

    email_id_list = email_ids_bytes[0].split()
    print(f"Found {len(email_id_list)} email(s) matching criteria.")

    parsed_emails_list = []
    processed_count = 0

    for email_id_b in email_id_list:
        if processed_count >= max_emails_to_process:
            print(f"Reached processing limit of {max_emails_to_process} emails for this polling cycle.")
            break

        email_id_str = email_id_b.decode()
        status, msg_data = mail_server.fetch(email_id_b, "(RFC822)")
        if status != "OK": print(f"Error fetching email ID {email_id_str}"); continue

        # Mark email as SEEN (optional, depending on workflow)
        # mail_server.store(email_id_b, '+FLAGS', '\\Seen')

        processed_count += 1
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                msg = email.message_from_bytes(response_part[1])
                subject = "".join([p.decode(c if c else "utf-8", "r") if isinstance(p, bytes) else p for p, c in decode_header(msg["Subject"] or "")])
                from_name, from_email = parseaddr(msg.get("From", ""))
                from_email = from_email.lower()
                message_id_header = msg.get("Message-ID", "").strip("<>")
                doc_id_prefix_source = message_id_header if message_id_header else f"uid_{email_id_str}"

                body, html_body_for_urls = "", ""
                if msg.is_multipart():
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        if "attachment" not in str(part.get("Content-Disposition")):
                            if content_type == "text/plain" and not body:
                                try:
                                    body = part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8', errors='replace')
                                except Exception as e:
                                    print(f"Error decoding multipart text/plain: {e}")
                                    pass # body remains empty or as previously set
                            elif content_type == "text/html":
                                try:
                                    html_c = part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8', errors='replace')
                                    html_body_for_urls = html_c
                                    if not body: # Only use HTML for body if plain text isn't found or failed
                                        soup = BeautifulSoup(html_c, "html.parser")
                                        body = soup.get_text('\n', True)
                                except Exception as e:
                                    print(f"Error parsing multipart html: {e}")
                                    pass
                else: # Not multipart
                    content_type = msg.get_content_type()
                    payload = msg.get_payload(decode=True)
                    charset = msg.get_content_charset() or 'utf-8'
                    if content_type == "text/plain":
                        try:
                            body = payload.decode(charset, errors='replace')
                        except Exception as e:
                            print(f"Error decoding non-multipart text/plain: {e}")
                            pass # body remains empty
                    elif content_type == "text/html":
                        try:
                            html_body_for_urls = payload.decode(charset, errors='replace')
                            soup = BeautifulSoup(html_body_for_urls, "html.parser")
                            body = soup.get_text('\n', True)
                        except Exception as e:
                            print(f"Error parsing non-multipart html: {e}")
                            pass # body remains empty

                urls = sorted(list(set(extract_urls_from_text(body) + ([a['href'] for a in BeautifulSoup(html_body_for_urls, 'html.parser').find_all('a', href=True) if a['href'] and not a['href'].startswith('mailto:')] if html_body_for_urls else []))))

                crawled_items = []
                if urls:
                    for u_crawl in urls[:2]: # Limit crawling
                        c_text = fetch_url_content(u_crawl)
                        crawled_items.append({"url": u_crawl, "text_content": c_text, "doc_id": f"crawled_{sanitize_table_name(u_crawl)}_from_{doc_id_prefix_source}"})

                email_attachments = extract_attachments(msg.walk(), subject, doc_id_prefix_source)
                parsed_emails_list.append({
                    "email_uid": email_id_str, "message_id_header": message_id_header,
                    "doc_id_prefix": doc_id_prefix_source, "subject": subject,
                    "from_name": from_name, "from_email": from_email, "to": msg.get("To"), "date": msg.get("Date"),
                    "body": body.strip(), "extracted_urls": urls,
                    "crawled_content": crawled_items, "attachments": email_attachments
                })
    return parsed_emails_list

def process_and_store_emails(mail_server, default_user_id_if_no_sender: str = "shared_kompow_user", max_emails_to_process_cycle: int = 10) -> list[str]:
    parsed_emails = parse_email_data(mail_server, max_emails_to_process=max_emails_to_process_cycle)
    if not parsed_emails:
        # print("No emails were parsed in this cycle. Nothing to store in Knowledge Base.") # Less verbose for loop
        return []

    print(f"--- Processing {len(parsed_emails)} Parsed Email(s) for Knowledge Base Storage ---")
    updated_user_ids_set = set() # To store user_ids whose KBs were updated

    for email_data in parsed_emails:
        user_id = email_data.get('from_email')
        if not user_id or '@' not in user_id :
            user_id = default_user_id_if_no_sender
            # print(f"Sender email not found/invalid for email (Subj: '{email_data['subject'][:30]}...'). Using default user_id: {user_id}")

        # print(f"Processing email from '{user_id}' (Subj: {email_data['subject'][:50]}...) for KB.")
        kb = get_user_knowledge_base(user_id)
        if not kb:
            print(f"Could not get/create KB for user {user_id}. Skipping storage for this email.")
            continue

        doc_added_for_this_user_in_this_email = False
        # 1. Store Email Body
        if email_data['body']:
            body_doc_id = f"email_{email_data['doc_id_prefix']}_body"
            body_metadata = {'source': 'email_body', 'subject': email_data['subject'], 'email_date': email_data['date'], 'from': email_data['from_email'], 'message_id': email_data['message_id_header'], 'user_id': user_id}
            if add_document_to_kb(kb, email_data['body'], body_metadata, body_doc_id):
                doc_added_for_this_user_in_this_email = True

        # 2. Store Attachment Content
        for attachment in email_data['attachments']:
            if attachment['data'] and not attachment['data'].startswith(("Error parsing", "No text found", "Attachment type", "PDF attachment", "DOCX attachment")):
                att_metadata = {'source': 'email_attachment', 'filename': attachment['filename'], 'content_type': attachment['content_type'], 'email_subject': email_data['subject'], 'message_id': email_data['message_id_header'], 'user_id': user_id}
                if add_document_to_kb(kb, attachment['data'], att_metadata, attachment['doc_id_base']):
                    doc_added_for_this_user_in_this_email = True

        # 3. Store Crawled Web Content
        for crawled_item in email_data['crawled_content']:
            if crawled_item['text_content']:
                crawl_metadata = {'source': 'crawled_url', 'url': crawled_item['url'], 'email_subject_source': email_data['subject'], 'message_id': email_data['message_id_header'], 'user_id': user_id}
                if add_document_to_kb(kb, crawled_item['text_content'], crawl_metadata, crawled_item['doc_id']):
                    doc_added_for_this_user_in_this_email = True

        if doc_added_for_this_user_in_this_email:
            updated_user_ids_set.add(user_id)
            print(f"KB updated for user '{user_id}' based on email (Subj: {email_data['subject'][:50]}...).")
        # else: # Less verbose
            # print(f"No new content from email (Subj: {email_data['subject'][:50]}...) was added to KB for user '{user_id}'.")

    if updated_user_ids_set:
        print(f"Knowledge Bases updated in this cycle for users: {list(updated_user_ids_set)}")
    else:
        print("No user KBs were updated with new content in this cycle.")

    return list(updated_user_ids_set)


if __name__ == "__main__":
    # This main block is for testing email_parser.py functionality directly.
    # The main application loop is in main.py in the project root.
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    load_dotenv(dotenv_path=dotenv_path)

    email_host = os.getenv("EMAIL_HOST")
    email_user = os.getenv("EMAIL_USER")
    email_pass = os.getenv("EMAIL_PASS")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    print("--- Direct Test of email_parser.py ---")
    if not all([email_host, email_user, email_pass]) or \
       email_host == "your_imap_server.com" or \
       email_user == "your_email@example.com":
        print("Email credentials in .env are placeholders or missing. Skipping live email fetching test.")
    elif not openai_api_key or openai_api_key == "your_openai_api_key_here":
        print("OPENAI_API_KEY is not set or is a placeholder. KB operations (add_document_to_kb) will fail if they require embeddings.")
        # Test can still proceed to see parsing, but KB storage will be affected.
        mail_server_conn = connect_to_mailbox(email_host, email_user, email_pass)
        if mail_server_conn:
            print("Connected to mailbox (API key missing, KB ops might fail).")
            updated_users = process_and_store_emails(mail_server_conn, default_user_id_if_no_sender=email_user, max_emails_to_process_cycle=2)
            print(f"Users whose KBs were attempted to be updated: {updated_users}")
            mail_server_conn.logout()
            print("Disconnected from mailbox.")
    else:
        print("Credentials and API key seem to be present. Attempting full email processing and KB storage test.")
        mail_server_conn = connect_to_mailbox(email_host, email_user, email_pass)
        if mail_server_conn:
            updated_users = process_and_store_emails(mail_server_conn, default_user_id_if_no_sender=email_user, max_emails_to_process_cycle=2) # Process 2 emails for test
            print(f"Users whose KBs were updated: {updated_users}")
            mail_server_conn.logout()
            print("Disconnected from mailbox.")

    print("\n--- email_parser.py Test Run Complete ---")
