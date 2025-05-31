import unittest
from unittest.mock import patch, MagicMock
import os
import sys
from email.message import EmailMessage
from email.header import Header # Import Header for robust encoding test
from bs4 import BeautifulSoup

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.email_parser import extract_urls_from_text, decode_filename

class TestEmailParserHelpers(unittest.TestCase):

    def test_extract_urls_from_text_plain(self):
        text_1 = "Visit http://example.com or https://www.google.com for more info. Also check www.another-site.net"
        expected_1 = sorted(["http://example.com", "https://www.google.com", "http://www.another-site.net"])
        self.assertEqual(sorted(extract_urls_from_text(text_1)), expected_1)

        text_2 = "No URLs here."
        expected_2 = []
        self.assertEqual(extract_urls_from_text(text_2), expected_2)

        text_3 = "A malformed url: http//oops.com and a good one: ftp://files.example.com (ftp not extracted by current regex)"
        expected_3 = []
        self.assertEqual(sorted(extract_urls_from_text(text_3)), expected_3)

        text_4 = "URL with path: https://example.com/path/to/page?query=true"
        expected_4 = ["https://example.com/path/to/page?query=true"]
        self.assertEqual(sorted(extract_urls_from_text(text_4)), expected_4)

        text_5 = "Multiple occurrences: http://site.com and http://site.com again."
        expected_5 = ["http://site.com"]
        self.assertEqual(sorted(extract_urls_from_text(text_5)), expected_5)

        text_6 = "Email link user@example.com should not be extracted as http."
        expected_6 = []
        self.assertEqual(extract_urls_from_text(text_6), expected_6)

    def test_decode_filename_rfc2047(self):
        expected_decoded_correct = "Filenamé è àçã.txt"
        # Use Python's Header class to correctly encode it according to RFC 2047
        h = Header(expected_decoded_correct, 'utf-8')
        encoded_filename_correct = h.encode()
        # This will produce something like "=?utf-8?q?Filenam=C3=A9=20=C3=A8=20=C3=A0=C3=A7=C3=A3.txt?="
        # or base64 if it's shorter. The actual format doesn't matter as long as it's standard.

        self.assertEqual(decode_filename(encoded_filename_correct), expected_decoded_correct)

        plain_filename = "simple_filename.pdf"
        self.assertEqual(decode_filename(plain_filename), plain_filename)

        empty_filename = ""
        self.assertEqual(decode_filename(empty_filename), empty_filename)

        none_filename = None
        self.assertIsNone(decode_filename(none_filename))

        mixed_encoding = "=?UTF-8?Q?Report_for_Q1?= (=?UTF-8?Q?=E2=82=AC_Symbol?=).docx"
        expected_mixed = "Report for Q1 (€ Symbol).docx"
        self.assertEqual(decode_filename(mixed_encoding), expected_mixed)


class TestEmailBodyParsing(unittest.TestCase):

    def _create_text_part(self, payload_str, content_subtype, charset='utf-8'):
        part = EmailMessage()
        part.set_content(payload_str, subtype=content_subtype, charset=charset)
        return part

    def test_get_body_from_plain_text_message(self):
        msg = EmailMessage()
        plain_content = "This is a plain text email body."
        msg.set_content(plain_content)

        body = ""
        if not msg.is_multipart():
            payload = msg.get_payload(decode=True)
            charset = msg.get_content_charset() or 'utf-8'
            body = payload.decode(charset, errors='replace').strip()

        self.assertEqual(body, plain_content)


    def test_get_body_from_html_message(self):
        html_content_raw = "<html><body><h1>Hello</h1><p>This is HTML.</p><a href='http://example.com'>Link</a></body></html>"
        msg = EmailMessage()
        msg.set_content(html_content_raw, subtype='html')

        body = ""
        html_body_for_urls = ""
        if not msg.is_multipart():
            payload = msg.get_payload(decode=True)
            charset = msg.get_content_charset() or 'utf-8'
            html_body_for_urls_decoded = payload.decode(charset, errors='replace')
            html_body_for_urls = html_body_for_urls_decoded.strip()

            soup = BeautifulSoup(html_body_for_urls_decoded, "html.parser")
            body = soup.get_text('\n', True)

        self.assertEqual(html_body_for_urls, html_content_raw)
        self.assertEqual(body, "Hello\nThis is HTML.\nLink")


    def test_get_body_multipart_prefer_plain(self):
        msg = EmailMessage()
        msg.make_alternative()

        plain_text_content = "This is the plain text part."
        html_content = "<html><body><p>This is HTML.</p></body></html>"

        plain_part = self._create_text_part(plain_text_content, "plain")
        html_part = self._create_text_part(html_content, "html")

        msg.attach(plain_part)
        msg.attach(html_part)

        body = ""
        html_body_for_urls_parsed = ""
        if msg.is_multipart():
            for part_iter in msg.walk():
                content_type = part_iter.get_content_type()
                if "attachment" not in str(part_iter.get("Content-Disposition")):
                    if content_type == "text/plain" and not body:
                        payload_bytes = part_iter.get_payload(decode=True)
                        charset = part_iter.get_content_charset() or 'utf-8'
                        body = payload_bytes.decode(charset, errors='replace')
                    elif content_type == "text/html":
                        payload_bytes = part_iter.get_payload(decode=True)
                        charset = part_iter.get_content_charset() or 'utf-8'
                        html_c = payload_bytes.decode(charset, errors='replace')
                        html_body_for_urls_parsed = html_c
                        if not body:
                            soup = BeautifulSoup(html_c, "html.parser")
                            body = soup.get_text('\n', True)

        self.assertEqual(body.strip(), plain_text_content)
        self.assertEqual(html_body_for_urls_parsed.strip(), html_content)


    def test_get_body_multipart_html_only_text_parts(self):
        msg = EmailMessage()
        msg.make_alternative()
        html_content = "<html><body><p>Only HTML here.</p></body></html>"

        html_part = self._create_text_part(html_content, "html")
        msg.attach(html_part)

        body = ""
        html_body_for_urls_parsed = ""
        if msg.is_multipart():
            for part_iter in msg.walk():
                content_type = part_iter.get_content_type()
                if "attachment" not in str(part_iter.get("Content-Disposition")):
                    if content_type == "text/plain" and not body:
                        payload_bytes = part_iter.get_payload(decode=True)
                        charset = part_iter.get_content_charset() or 'utf-8'
                        body = payload_bytes.decode(charset, errors='replace')
                    elif content_type == "text/html":
                        payload_bytes = part_iter.get_payload(decode=True)
                        charset = part_iter.get_content_charset() or 'utf-8'
                        html_c = payload_bytes.decode(charset, errors='replace')
                        html_body_for_urls_parsed = html_c
                        if not body:
                            soup = BeautifulSoup(html_c, "html.parser")
                            body = soup.get_text('\n', True)

        self.assertEqual(body.strip(), "Only HTML here.")
        self.assertEqual(html_body_for_urls_parsed.strip(), html_content)

    def test_get_body_no_text_payload(self):
        msg = EmailMessage()
        msg.make_mixed()

        image_part = EmailMessage()
        image_part.set_payload(b"imagedata_bytes_here")
        image_part.set_type("image/jpeg")
        image_part['Content-Transfer-Encoding'] = 'base64'
        image_part.add_header('Content-Disposition', 'inline', filename='image.jpg')
        msg.attach(image_part)

        body = ""
        html_body_for_urls = ""
        if msg.is_multipart():
            for part_iter in msg.walk():
                content_type = part_iter.get_content_type()
                if "attachment" not in str(part_iter.get("Content-Disposition")):
                    if content_type == "text/plain" and not body:
                        payload_bytes = part_iter.get_payload(decode=True)
                        charset = part_iter.get_content_charset() or 'utf-8'
                        body = payload_bytes.decode(charset, errors='replace')
                    elif content_type == "text/html":
                        payload_bytes = part_iter.get_payload(decode=True)
                        charset = part_iter.get_content_charset() or 'utf-8'
                        html_c = payload_bytes.decode(charset, errors='replace')
                        html_body_for_urls = html_c
                        if not body:
                             soup = BeautifulSoup(html_c, "html.parser")
                             body = soup.get_text('\n', True)

        self.assertEqual(body, "")
        self.assertEqual(html_body_for_urls, "")

if __name__ == '__main__':
    unittest.main()
