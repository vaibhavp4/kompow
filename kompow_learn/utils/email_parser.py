# This file will contain utility functions for parsing emails.
import imaplib
import email
from email.header import decode_header
from email.utils import parseaddr # For parsing 'From' field
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import re
import io

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None
    print("pypdf not found. PDF attachment parsing will be skipped. Please install it.")

try:
    import docx
except ImportError:
    docx = None
    print("python-docx not found. DOCX attachment parsing will be skipped. Please install it.")

from .web_crawler import fetch_url_content
from .knowledge_base import get_user_knowledge_base, add_document_to_kb, query_knowledge_base, sanitize_table_name


def connect_to_mailbox(host, user, password):
    """Connects to the IMAP server and returns the mail server connection object."""
    try:
        mail = imaplib.IMAP4_SSL(host)
        mail.login(user, password)
        print("Successfully connected to the mailbox.")
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
                    try:
                        attachment_data_text = payload.decode('utf-8', errors='replace')
                        # print(f"Extracted text from attachment: {filename}")
                    except Exception as e:
                        print(f"Could not decode .txt attachment {filename}: {e}")
                        attachment_data_text = f"Error decoding .txt attachment: {e}"
                elif file_ext == "pdf":
                    if PdfReader:
                        try:
                            pdf_stream = io.BytesIO(payload)
                            reader = PdfReader(pdf_stream)
                            text = "".join([reader.pages[p].extract_text() or "" for p in range(len(reader.pages))])
                            attachment_data_text = text.strip() if text else "No text found in PDF."
                            # print(f"Extracted text from PDF: {filename}" if text.strip() else f"No text found in PDF: {filename}")
                        except Exception as e:
                            print(f"Error parsing PDF attachment {filename}: {e}")
                            attachment_data_text = f"Error parsing PDF: {e}"
                    else:
                        attachment_data_text = f"PDF attachment: {filename} (pypdf not installed)"
                elif file_ext == "docx":
                    if docx:
                        try:
                            docx_stream = io.BytesIO(payload)
                            document = docx.Document(docx_stream)
                            attachment_data_text = "\n".join([para.text for para in document.paragraphs]).strip()
                            # print(f"Extracted text from DOCX: {filename}" if attachment_data_text else f"No text found in DOCX: {filename}")
                        except Exception as e:
                            print(f"Error parsing DOCX attachment {filename}: {e}")
                            attachment_data_text = f"Error parsing DOCX: {e}"
                    else:
                        attachment_data_text = f"DOCX attachment: {filename} (python-docx not installed)"
                else:
                    attachment_data_text = f"Attachment type '{file_ext}' not parsed: {filename}"

                attachments.append({
                    "filename": filename,
                    "content_type": content_type,
                    "data": attachment_data_text, # This now holds the extracted text or error/placeholder
                    "doc_id_base": f"email_{email_id_for_doc_id}_attachment_{sanitize_table_name(filename)}"
                })
    return attachments

def parse_email_data(mail_server):
    """
    Internal function to fetch and parse emails into a structured format.
    Returns a list of dictionaries, each representing an email.
    """
    if mail_server is None: return []
    status, _ = mail_server.select("INBOX")
    if status != "OK": print("Error selecting INBOX"); return []
    status, email_ids_bytes = mail_server.search(None, "ALL") # Fetch all for testing
    if status != "OK": print("Error searching for emails"); return []

    email_id_list = email_ids_bytes[0].split()
    print(f"Found {len(email_id_list)} emails to parse.")

    parsed_emails_list = []

    for email_id_b in email_id_list[:5]: # Process only first 5 emails for testing speed
        email_id_str = email_id_b.decode()
        status, msg_data = mail_server.fetch(email_id_b, "(RFC822)")
        if status != "OK": print(f"Error fetching email ID {email_id_str}"); continue

        for response_part in msg_data:
            if isinstance(response_part, tuple):
                msg = email.message_from_bytes(response_part[1])

                subject_header = decode_header(msg["Subject"])
                subject = "".join([p.decode(c if c else "utf-8", "r") if isinstance(p, bytes) else p for p, c in subject_header])

                from_header = msg.get("From", "")
                from_name, from_email = parseaddr(from_header)
                from_email = from_email.lower() # Normalize email address

                to_ = msg.get("To")
                date_ = msg.get("Date")
                message_id_header = msg.get("Message-ID", "").strip("<>")

                # Use Message-ID for a more unique doc_id base if available, else email_id
                doc_id_prefix_source = message_id_header if message_id_header else email_id_str

                body, html_body_for_urls = "", ""
                if msg.is_multipart():
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        if "attachment" not in str(part.get("Content-Disposition")):
                            if content_type == "text/plain" and not body:
                                try: body = part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8', 'r')
                                except: pass
                            elif content_type == "text/html":
                                try:
                                    html_c = part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8', 'r')
                                    html_body_for_urls = html_c
                                    if not body: soup = BeautifulSoup(html_c, "html.parser"); body = soup.get_text('\n', True)
                                except: pass
                else: # Not multipart
                    content_type = msg.get_content_type()
                    payload = msg.get_payload(decode=True)
                    charset = msg.get_content_charset() or 'utf-8'
                    if content_type == "text/plain":
                        try: body = payload.decode(charset, 'r')
                        except: pass
                    elif content_type == "text/html":
                        html_body_for_urls = payload.decode(charset, 'r')
                        try: soup = BeautifulSoup(html_body_for_urls, "html.parser"); body = soup.get_text('\n', True)
                        except: pass

                urls_body = extract_urls_from_text(body)
                urls_html = []
                if html_body_for_urls:
                    soup_urls = BeautifulSoup(html_body_for_urls, 'html.parser')
                    urls_html = [a['href'] for a in soup_urls.find_all('a', href=True) if a['href'] and not a['href'].startswith('mailto:')]
                all_urls = sorted(list(set(urls_body + urls_html)))

                crawled_items = []
                if all_urls:
                    # print(f"Found {len(all_urls)} URLs in email '{subject[:30]}...'. Crawling up to 2.")
                    for idx, u_crawl in enumerate(all_urls[:2]): # Limit crawling to 2 URLs per email for now
                        # print(f"  Crawling URL {idx+1}/{len(all_urls)}: {u_crawl}")
                        c_text = fetch_url_content(u_crawl)
                        doc_id_crawl = f"crawled_{sanitize_table_name(u_crawl)}_from_email_{doc_id_prefix_source}"
                        crawled_items.append({"url": u_crawl, "text_content": c_text, "doc_id": doc_id_crawl})

                email_attachments = extract_attachments(msg.walk(), subject, doc_id_prefix_source)

                parsed_emails_list.append({
                    "email_uid": email_id_str, # UID from IMAP server for this mailbox session
                    "message_id_header": message_id_header, # Message-ID header
                    "doc_id_prefix": doc_id_prefix_source, # Base for document IDs from this email
                    "subject": subject, "from_name": from_name, "from_email": from_email,
                    "to": to_, "date": date_, "body": body.strip(),
                    "extracted_urls": all_urls, "crawled_content": crawled_items,
                    "attachments": email_attachments
                })
    return parsed_emails_list

def process_and_store_emails(mail_server, default_user_id_if_no_sender: str = "shared_kompow_user"):
    """
    Fetches emails, parses them, and stores relevant content into user-specific knowledge bases.
    """
    parsed_emails = parse_email_data(mail_server)
    if not parsed_emails:
        print("No emails were parsed. Nothing to store in Knowledge Base.")
        return

    print(f"\n--- Processing {len(parsed_emails)} Emails for Knowledge Base Storage ---")
    for email_data in parsed_emails:
        user_id = email_data.get('from_email')
        if not user_id or '@' not in user_id : # Basic check for valid email
            print(f"Sender email not found or invalid for email subject: '{email_data['subject']}'. Using default user_id: {default_user_id_if_no_sender}")
            user_id = default_user_id_if_no_sender
        else:
            print(f"Processing email from '{user_id}' (Subject: {email_data['subject'][:50]}...) for Knowledge Base.")

        kb = get_user_knowledge_base(user_id)
        if not kb:
            print(f"Could not get/create Knowledge Base for user {user_id}. Skipping storage for this email.")
            continue

        # 1. Store Email Body
        if email_data['body']:
            body_doc_id = f"email_{email_data['doc_id_prefix']}_body"
            body_metadata = {
                'source': 'email_body', 'subject': email_data['subject'],
                'email_date': email_data['date'], 'from': email_data['from_email'],
                'message_id': email_data['message_id_header']
            }
            print(f"  Adding email body to KB for {user_id}. Doc ID: {body_doc_id}")
            add_document_to_kb(kb, email_data['body'], body_metadata, body_doc_id)

        # 2. Store Attachment Content
        for attachment in email_data['attachments']:
            # Check if data is actual content and not an error message or placeholder
            if attachment['data'] and not attachment['data'].startswith(("Error parsing", "No text found", "Attachment type", "PDF attachment", "DOCX attachment")):
                att_metadata = {
                    'source': 'email_attachment', 'filename': attachment['filename'],
                    'content_type': attachment['content_type'], 'email_subject': email_data['subject'],
                    'message_id': email_data['message_id_header']
                }
                print(f"  Adding attachment '{attachment['filename']}' to KB for {user_id}. Doc ID: {attachment['doc_id_base']}")
                add_document_to_kb(kb, attachment['data'], att_metadata, attachment['doc_id_base'])

        # 3. Store Crawled Web Content
        for crawled_item in email_data['crawled_content']:
            if crawled_item['text_content']:
                crawl_metadata = {
                    'source': 'crawled_url', 'url': crawled_item['url'],
                    'email_subject_source': email_data['subject'], # Subject of email where URL was found
                    'message_id': email_data['message_id_header']
                }
                print(f"  Adding crawled content from '{crawled_item['url']}' to KB for {user_id}. Doc ID: {crawled_item['doc_id']}")
                add_document_to_kb(kb, crawled_item['text_content'], crawl_metadata, crawled_item['doc_id'])
        print(f"--- Finished processing email (Subject: {email_data['subject'][:50]}...) for user {user_id} ---")


if __name__ == "__main__":
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    load_dotenv(dotenv_path=dotenv_path)

    email_host = os.getenv("EMAIL_HOST")
    email_user = os.getenv("EMAIL_USER") # This user's mailbox will be read
    email_pass = os.getenv("EMAIL_PASS")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    print("--- Email Parser & KB Integration Test ---")
    if not all([email_host, email_user, email_pass]) or \
       email_host == "your_imap_server.com" or \
       email_user == "your_email@example.com":
        print("Email credentials in .env are placeholders or missing. Skipping live email fetching.")
        print("To run a full test, configure your IMAP server details and credentials in .env.")
    elif not openai_api_key or openai_api_key == "your_openai_api_key_here":
        print("OPENAI_API_KEY is not set or is a placeholder in .env.")
        print("Knowledge Base operations (adding, searching) will likely fail or use dummy embeddings.")
        print("Set a valid OPENAI_API_KEY for full KB functionality.")
        # Optionally, could still run mail_server connection and parsing without KB ops
        mail_server_conn = connect_to_mailbox(email_host, email_user, email_pass)
        if mail_server_conn:
            process_and_store_emails(mail_server_conn) # Will attempt KB, but embeddings will fail
            mail_server_conn.logout()
            print("Disconnected from mailbox.")
    else:
        print("Credentials and API key seem to be present. Attempting full processing.")
        mail_server_conn = connect_to_mailbox(email_host, email_user, email_pass)
        if mail_server_conn:
            process_and_store_emails(mail_server_conn, default_user_id_if_no_sender=email_user) # Use logged-in user as default
            mail_server_conn.logout()
            print("Disconnected from mailbox.")

            # Optional: Test query for one of the users processed if emails were found
            # This requires knowing/guessing a user_id that was processed.
            # For example, if you know an email was sent by "test_sender@example.com":
            # test_query_user = "test_sender@example.com"
            # kb_for_query = get_user_knowledge_base(test_query_user)
            # if kb_for_query:
            #     print(f"\n--- Querying KB for user {test_query_user} ---")
            #     sample_query = "python" # replace with a relevant term
            #     results = query_knowledge_base(kb_for_query, sample_query, limit=2)
            #     if results:
            #         for res in results:
            #             print(f"  Found Doc ID: {res.id}, Content: {res.content[:100]}..., Meta: {res.metadata}")
            #     else:
            #         print(f"No results found for query '{sample_query}' for user {test_query_user}.")
            # else:
            #     print(f"Could not get KB for user {test_query_user} to test query.")
    print("\n--- Test Run Complete ---")
