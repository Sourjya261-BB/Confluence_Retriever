import os
import requests
import markdownify
from urllib.parse import urljoin
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

load_dotenv() 

# Confluence API endpoint and credentials
CONFLUENCE_URL = "https://bigbasket.atlassian.net/wiki"
SPACE_KEY = 'BIG'
USERNAME = os.environ.get("USERNAME")
API_TOKEN = os.environ.get("API_TOKEN")
auth = HTTPBasicAuth(USERNAME, API_TOKEN)

# Headers for authentication
HEADERS = {
    'Content-Type': 'application/json'
}

# Output directory
OUTPUT_DIR = '/home/sourjyamukherjee/Documents/confluence_retriever/md_output'

def get_pages(space_key, start=0, limit=25):
    url = f'{CONFLUENCE_URL}/rest/api/content'
    params = {
        'spaceKey': space_key,
        'start': start,
        'limit': limit,
        'expand': 'body.storage,version'
    }
    response = requests.get(url, headers=HEADERS, params=params, auth=auth)
    response.raise_for_status()
    return response.json()

def get_attachments(page_id):
    url = f'{CONFLUENCE_URL}/rest/api/content/{page_id}/child/attachment'
    response = requests.get(url, headers=HEADERS, auth=auth)
    response.raise_for_status()
    return response.json()

def sanitize_filename(filename):
    #return ''.join(c if c.isalnum() or c in (' ', '.', '_') else '_' for c in filename)
    filename = filename.replace('/', '_')
    return filename

def download_attachments(page_id, page_title):
    page_title = sanitize_filename(page_title)
    page_dir = os.path.join(OUTPUT_DIR, page_title)

    if not os.path.exists(page_dir):
        os.makedirs(page_dir, exist_ok=True)

    attachments = get_attachments(page_id)
    local_paths = []
    
    for attachment in attachments.get('results', []):
        attachment_url = CONFLUENCE_URL + attachment['_links']['download']
        attachment_name = attachment['title']
        attachment_path = os.path.join(page_dir, attachment_name)
        print(f"attachement_url:{attachment_url}")

        try:
            response = requests.get(attachment_url, auth=(USERNAME, API_TOKEN))
            response.raise_for_status()

            with open(attachment_path, 'wb') as file:
                file.write(response.content)
                print(f"Attachment found: {attachment_name}")

            local_paths.append(attachment_path)
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                print(f"Attachment not found: {attachment_name} (404)")
            else:
                raise e

    return local_paths

def save_page_as_markdown(page):
    title = page['title']
    page_id = page['id']
    content = page['body']['storage']['value']
    source_url = page['_links']['self']
    print(f"source_url:{source_url}")
    
    # Convert HTML content to Markdown
    markdown_content = markdownify.markdownify(content, heading_style="ATX")
    
    # Download attachments and prepare their paths and the source URL section
    attachments = download_attachments(page_id, title)
    attachments_section = f"\n### Attachments:\n" + "\n".join([f"- {path}" for path in attachments]) + "\n"
    source_section = f"\n### Source:\n{source_url}\n"
    
    # Save to Markdown file
    with open(os.path.join(OUTPUT_DIR, f'{sanitize_filename(title)}.md'), 'w', encoding='utf-8') as md_file:
        md_file.write(f"# {title}\n{attachments_section}{source_section}\n### Page_content:\n{markdown_content}")

def main():
    start = 0
    limit = 20
    while True:
        pages_data = get_pages(SPACE_KEY, start=start, limit=limit)
        pages = pages_data['results']
        for page in pages:
            save_page_as_markdown(page)
        
        if 'next' not in pages_data['_links']:
            break
        start += limit

if __name__ == '__main__':
    main()
