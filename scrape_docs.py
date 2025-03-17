import os
import requests
import markdownify
from urllib.parse import urljoin
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

load_dotenv(override=True) 

# Confluence API endpoint and credentials
CONFLUENCE_URL = "https://bigbasket.atlassian.net/wiki"
SPACE_KEY = 'BIG'
USERNAME = os.environ.get("USERNAME")
API_TOKEN = os.environ.get("API_TOKEN")
print(f"USERNAME:{USERNAME}, API_TOKEN:{API_TOKEN}")
auth = HTTPBasicAuth(USERNAME, API_TOKEN)

# Headers for authentication
HEADERS = {
    'Content-Type': 'application/json'
}

# Output directory
OUTPUT_DIR = './md_output'

def count_processed_pages():
    """ Count how many pages have already been processed """
    if os.path.exists(OUTPUT_DIR):
        return len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.md')])
    else:
        return 0

# def get_pages(space_key, start=0, limit=25):
#     url = f'{CONFLUENCE_URL}/rest/api/content'
#     params = {
#         'spaceKey': space_key,
#         'start': start,
#         'limit': limit,
#         'expand': 'body.storage,version'
#     }
#     response = requests.get(url, headers=HEADERS, params=params, auth=auth)
#     response.raise_for_status()
#     return response.json()

def get_pages(space_key, start=0, limit=25):
    url = f'{CONFLUENCE_URL}/rest/api/content'
    params = {
        'spaceKey': space_key,
        'start': start,
        'limit': limit,
        'expand': 'body.storage,version',
        'orderBy': 'version.lastUpdated DESC'  # -> addressed the orderig issue as of friday 28th feb
    }
    response = requests.get(url, headers=HEADERS, params=params, auth=auth)

    if response.status_code == 400:
        print(f"Bad Request: {response.text}")  # Print API response for debugging

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
    markdown_filename = f"{sanitize_filename(title)}.md"
    markdown_path = os.path.join(OUTPUT_DIR, markdown_filename)

    # Skip if the file already exists
    if os.path.exists(markdown_path):
        print(f"Skipping {title}, already processed.")
        return
    
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
    limit = 50
    page_count = 0
    max_pages=1000
    print(f"Processed Pages: {count_processed_pages()}")
    # while True:
    while page_count < max_pages:
        pages_data = get_pages(SPACE_KEY, start=start, limit=limit)
        pages = pages_data['results']
        for page in pages:
            save_page_as_markdown(page)
            page_count=page_count+1
        
        if 'next' not in pages_data['_links']:
            break
        start += limit

if __name__ == '__main__':
    main()




# import os
# import requests
# import markdownify
# from tqdm import tqdm
# from requests.auth import HTTPBasicAuth
# from dotenv import load_dotenv

# load_dotenv(override=True)

# # Confluence API endpoint and credentials
# CONFLUENCE_URL = "https://bigbasket.atlassian.net/wiki"
# SPACE_KEY = 'BIG'
# USERNAME = os.environ.get("USERNAME")
# API_TOKEN = os.environ.get("API_TOKEN")
# auth = HTTPBasicAuth(USERNAME, API_TOKEN)

# # Headers for authentication
# HEADERS = {'Content-Type': 'application/json'}

# # Output directory
# OUTPUT_DIR = './md_output'
# os.makedirs(OUTPUT_DIR, exist_ok=True)


# def get_attachments(page_id):
#     url = f'{CONFLUENCE_URL}/rest/api/content/{page_id}/child/attachment'
#     response = requests.get(url, headers=HEADERS, auth=auth)
#     response.raise_for_status()
#     return response.json()

# def download_attachments(page_id, page_title):
#     page_title = sanitize_filename(page_title)
#     page_dir = os.path.join(OUTPUT_DIR, page_title)

#     if not os.path.exists(page_dir):
#         os.makedirs(page_dir, exist_ok=True)

#     attachments = get_attachments(page_id)
#     local_paths = []
    
#     for attachment in attachments.get('results', []):
#         attachment_url = CONFLUENCE_URL + attachment['_links']['download']
#         attachment_name = attachment['title']
#         attachment_path = os.path.join(page_dir, attachment_name)
#         print(f"attachement_url:{attachment_url}")

#         try:
#             response = requests.get(attachment_url, auth=(USERNAME, API_TOKEN))
#             response.raise_for_status()

#             with open(attachment_path, 'wb') as file:
#                 file.write(response.content)
#                 print(f"Attachment found: {attachment_name}")

#             local_paths.append(attachment_path)
#         except requests.exceptions.HTTPError as e:
#             if response.status_code == 404:
#                 print(f"Attachment not found: {attachment_name} (404)")
#             else:
#                 raise e

#     return local_paths

# def get_all_page_ids(space_key):
#     """ Fetch all page IDs and titles from Confluence. """
#     page_ids = {}
#     start = 0
#     limit = 50

#     while True:
#         url = f'{CONFLUENCE_URL}/rest/api/content'
#         params = {'spaceKey': space_key, 'start': start, 'limit': limit}
#         response = requests.get(url, headers=HEADERS, params=params, auth=auth)
#         response.raise_for_status()
#         data = response.json()

#         for page in data['results']:
#             title = sanitize_filename(page['title'])
#             page_ids[page['id']] = title  # Store page_id -> title mapping

#         if 'next' not in data['_links']:
#             break

#         start += limit

#     return page_ids

# def count_processed_pages():
#     """ Get a set of already processed page filenames. """
#     return {f[:-3] for f in os.listdir(OUTPUT_DIR) if f.endswith('.md')}

# def get_page_content(page_id):
#     """ Fetch full page content from Confluence. """
#     url = f'{CONFLUENCE_URL}/rest/api/content/{page_id}?expand=body.storage'
#     response = requests.get(url, headers=HEADERS, auth=auth)
#     response.raise_for_status()
#     return response.json()

# def sanitize_filename(filename):
#     """ Replace special characters to create a safe filename. """
#     return filename.replace('/', '_')

# def save_page_as_markdown(page_id,page_title, page_content, source_url):

#     print(f"source_url:{source_url}")
    
#     # Convert HTML content to Markdown
#     markdown_content = markdownify.markdownify(page_content, heading_style="ATX")
    
#     # Download attachments and prepare their paths and the source URL section
#     attachments = download_attachments(page_id, page_title)
#     attachments_section = f"\n### Attachments:\n" + "\n".join([f"- {path}" for path in attachments]) + "\n"
#     source_section = f"\n### Source:\n{source_url}\n"
    
#     # Save to Markdown file
#     with open(os.path.join(OUTPUT_DIR, f'{sanitize_filename(page_title)}.md'), 'w', encoding='utf-8') as md_file:
#         md_file.write(f"# {page_title}\n{attachments_section}{source_section}\n### Page_content:\n{markdown_content}")


# def main():
#     # Step 1: Get all pages and already processed pages
#     all_pages = get_all_page_ids(SPACE_KEY)
#     processed_pages = count_processed_pages()

#     # Step 2: Identify pages to fetch
#     unprocessed_pages = {id: title for id, title in all_pages.items() if title not in processed_pages}

#     print(f"Total Pages: {len(all_pages)}, Already Processed: {len(processed_pages)}, Remaining: {len(unprocessed_pages)}")

#     # Step 3: Process only unprocessed pages with a progress bar
#     progress_bar = tqdm(total=len(unprocessed_pages), desc="Processing Pages", unit="page")

#     for page_id, title in unprocessed_pages.items():
#         page_data = get_page_content(page_id)
#         page_content = page_data['body']['storage']['value']
#         source_url = page_data['_links']['self']
#         save_page_as_markdown(page_id,title, page_content, source_url)
#         progress_bar.update(1)

#     progress_bar.close()
#     print("Processing complete!")

# if __name__ == '__main__':
#     main()
