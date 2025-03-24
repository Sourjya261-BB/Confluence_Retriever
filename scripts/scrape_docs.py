import os
import requests
import markdownify
from urllib.parse import urljoin
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
import concurrent.futures
import time

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
# Number of worker threads for parallel processing
MAX_WORKERS = 10

def count_processed_pages():
    """ Count how many pages have already been processed """
    if os.path.exists(OUTPUT_DIR):
        return len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.md')])
    else:
        return 0

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
    filename = filename.replace('/', '_')
    return filename

def download_attachments(page_id, page_title):
    page_title = sanitize_filename(page_title)
    page_dir = os.path.join(OUTPUT_DIR, page_title)

    if not os.path.exists(page_dir):
        os.makedirs(page_dir, exist_ok=True)

    try:
        attachments = get_attachments(page_id)
    except requests.exceptions.HTTPError as e:
        print(f"Error getting attachments for page {page_id}: {str(e)}")
        return []
        
    local_paths = []
    
    for attachment in attachments.get('results', []):
        attachment_url = CONFLUENCE_URL + attachment['_links']['download']
        attachment_name = attachment['title']

        if attachment_name.lower().endswith(('.mp3', '.mp4')): #there are mp3 files also damnn
            print(f"Skipping attachment: {attachment_name} (MP3/MP4 file)")
            continue 

        attachment_path = os.path.join(page_dir, attachment_name)
        print(f"Downloading attachment: {attachment_url}")

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
            elif response.status_code == 400:
                print(f"Bad request for attachment: {attachment_name} (400) - Skipping")
                if "UNKNOWN_MEDIA_ID" in str(e):
                    print(f"  Attachment has UNKNOWN_MEDIA_ID issue")
            else:
                print(f"HTTP error downloading {attachment_name}: {str(e)}")
        except Exception as e:
            print(f"Error downloading {attachment_name}: {str(e)}")

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
    print(f"Processing page: {title} (ID: {page_id})")
    
    # Convert HTML content to Markdown
    markdown_content = markdownify.markdownify(content, heading_style="ATX")
    
    # Download attachments and prepare their paths and the source URL section
    attachments = download_attachments(page_id, title)
    attachments_section = f"\n### Attachments:\n" + "\n".join([f"- {path}" for path in attachments]) + "\n"
    source_section = f"\n### Source:\n{source_url}\n"
    
    # Save to Markdown file
    with open(os.path.join(OUTPUT_DIR, f'{sanitize_filename(title)}.md'), 'w', encoding='utf-8') as md_file:
        md_file.write(f"# {title}\n{attachments_section}{source_section}\n### Page_content:\n{markdown_content}")

def process_page(page):
    """Process a single page - for parallel execution"""
    try:
        title = page['title']
        page_id = page['id']
        markdown_filename = f"{sanitize_filename(title)}.md"
        markdown_path = os.path.join(OUTPUT_DIR, markdown_filename)

        # Skip if the file already exists
        if os.path.exists(markdown_path):
            print(f"Skipping {title}, already processed.")
            return title, "skipped"
        
        content = page['body']['storage']['value']
        source_url = page['_links']['self']
        print(f"Processing page: {title} (ID: {page_id})")
        
        # Convert HTML content to Markdown
        markdown_content = markdownify.markdownify(content, heading_style="ATX")
        
        # Download attachments and prepare their paths and the source URL section
        attachments = download_attachments(page_id, title)
        attachments_section = f"\n### Attachments:\n" + "\n".join([f"- {path}" for path in attachments]) + "\n"
        source_section = f"\n### Source:\n{source_url}\n"
        
        # Save to Markdown file
        with open(markdown_path, 'w', encoding='utf-8') as md_file:
            md_file.write(f"# {title}\n{attachments_section}{source_section}\n### Page_content:\n{markdown_content}")
        
        return title, True
    except Exception as e:
        return page.get('title', 'Unknown page'), f"Error: {str(e)}"

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
    start = 0
    limit = 150
    total_pages_processed = 0
    failed_pages = []
    
    print(f"Previously Processed Pages: {count_processed_pages()}")
    start_time = time.time()
    
    while True:
        try:
            pages_data = get_pages(SPACE_KEY, start=start, limit=limit)
            pages = pages_data['results']
            
            if not pages:
                print("No more pages to process.")
                break
                
            print(f"Processing batch of {len(pages)} pages starting from index {start}")
                
            # Process pages in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit all page processing tasks to the executor
                future_to_page = {executor.submit(process_page, page): page for page in pages}
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_page):
                    page = future_to_page[future]
                    try:
                        title, result = future.result()
                        if result is True:
                            print(f"✓ Successfully processed: {title}")
                            total_pages_processed += 1
                        elif result == "skipped":
                            print(f"↷ Skipped already processed: {title}")
                        else:
                            print(f"✗ Failed to process: {title} - {result}")
                            failed_pages.append((title, result))
                    except Exception as exc:
                        page_title = page.get('title', 'Unknown')
                        print(f"✗ Error processing page {page_title}: {exc}")
                        failed_pages.append((page_title, str(exc)))
            
            if 'next' not in pages_data['_links']:
                print("No more pages available.")
                break
            start += limit
            
        except Exception as e:
            print(f"Error in main processing loop: {str(e)}")
            break
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n===== SUMMARY =====")
    print(f"Completed processing {total_pages_processed} pages in {elapsed_time:.2f} seconds")
    if total_pages_processed > 0:
        print(f"Average time per page: {elapsed_time/total_pages_processed:.2f} seconds")
    
    if failed_pages:
        print(f"\nFailed to process {len(failed_pages)} pages:")
        for title, error in failed_pages[:20]:  # Show only first 20 failures to avoid overwhelming output
            print(f"- {title}: {error}")
        if len(failed_pages) > 20:
            print(f"  ...and {len(failed_pages) - 20} more failures")

if __name__ == '__main__':
    main()