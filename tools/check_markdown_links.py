import os
import re
import sys

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    markdown_files = []
    
    for root, _, files in os.walk(repo_root):
        if '.git' in root:
            continue
        for file in files:
            if file.endswith('.md'):
                markdown_files.append(os.path.join(root, file))
                
    image_regex = re.compile(r'!\[.*?\]\((.*?)\)')
    has_errors = False
    
    for md_file in markdown_files:
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        matches = image_regex.findall(content)
        for match in matches:
            # Ignore external URLs
            if match.startswith('http://') or match.startswith('https://'):
                continue
                
            md_dir = os.path.dirname(md_file)
            image_path = os.path.normpath(os.path.join(md_dir, match))
            
            if not os.path.exists(image_path):
                print(f"ERROR: Broken image link in {os.path.relpath(md_file, repo_root)}")
                print(f"       Cannot find image: {match} (resolved to {os.path.relpath(image_path, repo_root)})")
                has_errors = True
                
    if has_errors:
        print("\nPre-commit hook failed: Broken image links detected. Please fix them before committing.")
        sys.exit(1)
    else:
        print("Pre-commit hook passed: All markdown image links are valid.")
        sys.exit(0)

if __name__ == '__main__':
    main()
