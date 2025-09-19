# version 0.0.1 by romangorbunov91
# 19-Sep-2025

import numpy as np
import re

def update_readme_section(tbl_df, readme_path, tbl_name, section):
    markdown_table = tbl_df.to_markdown(index=False)

    with open(readme_path, 'r', encoding="utf-8-sig") as f:
        content = f.read()

    # Define start and end markers.
    start_marker = f'<!-- START_{section.upper()} -->'
    end_marker = f'<!-- END_{section.upper()} -->'

    # Wrap the table with headers and markers.
    new_section = f'\n{start_marker} \n### {tbl_name}\n{markdown_table}\n{end_marker}'
    
    # Remove any previous content between the markers.
    pattern = re.compile(f'{re.escape(start_marker)}.*?{re.escape(end_marker)}', re.DOTALL)
    updated_content = pattern.sub("", content)

    # Insert the new section to the end of the file.
    updated_content += new_section

    with open(readme_path, 'w', encoding="utf-8-sig") as f:
        f.write(updated_content)
