#!/usr/bin/env python3
"""
Simple Markdown to PDF converter using weasyprint
"""

import markdown
import weasyprint
import os
import sys

def convert_markdown_to_pdf(markdown_file, output_pdf=None):
    """Convert a Markdown file to PDF with proper styling"""
    
    if not os.path.exists(markdown_file):
        print(f"Error: File {markdown_file} does not exist")
        return False
    
    if output_pdf is None:
        output_pdf = os.path.splitext(markdown_file)[0] + '.pdf'
    
    # Read the markdown file
    with open(markdown_file, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Convert markdown to HTML
    md = markdown.Markdown(extensions=['tables', 'fenced_code', 'toc'])
    html_content = md.convert(markdown_content)
    
    # Create a complete HTML document with CSS styling
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Technical Report</title>
        <style>
            body {{
                font-family: 'DejaVu Sans', Arial, sans-serif;
                line-height: 1.6;
                margin: 40px;
                color: #333;
                font-size: 11pt;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                font-size: 24pt;
                margin-top: 30px;
            }}
            h2 {{
                color: #34495e;
                border-bottom: 2px solid #bdc3c7;
                padding-bottom: 5px;
                font-size: 18pt;
                margin-top: 25px;
            }}
            h3 {{
                color: #34495e;
                font-size: 14pt;
                margin-top: 20px;
            }}
            h4 {{
                color: #34495e;
                font-size: 12pt;
                margin-top: 15px;
            }}
            code {{
                background-color: #f8f9fa;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: 'DejaVu Sans Mono', monospace;
                font-size: 10pt;
            }}
            pre {{
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #3498db;
                overflow-x: auto;
                font-family: 'DejaVu Sans Mono', monospace;
                font-size: 9pt;
                line-height: 1.4;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 15px 0;
                font-size: 10pt;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            ul, ol {{
                margin: 10px 0;
                padding-left: 25px;
            }}
            li {{
                margin: 5px 0;
            }}
            blockquote {{
                border-left: 4px solid #3498db;
                margin: 15px 0;
                padding-left: 15px;
                font-style: italic;
                color: #666;
            }}
            .page-break {{
                page-break-before: always;
            }}
            @page {{
                margin: 2cm;
                @bottom-center {{
                    content: counter(page);
                }}
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    try:
        # Convert HTML to PDF
        html_doc = weasyprint.HTML(string=html_template)
        html_doc.write_pdf(output_pdf)
        print(f"✅ Successfully converted {markdown_file} to {output_pdf}")
        
        # Auto-open in VS Code if possible
        import subprocess
        try:
            subprocess.run(['code', output_pdf], check=False, capture_output=True)
            print(f"📖 Opened {output_pdf} in VS Code")
        except:
            pass
            
        return True
    except Exception as e:
        print(f"❌ Error converting to PDF: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_to_pdf.py <markdown_file> [output_pdf]")
        sys.exit(1)
    
    markdown_file = sys.argv[1]
    output_pdf = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = convert_markdown_to_pdf(markdown_file, output_pdf)
    sys.exit(0 if success else 1)