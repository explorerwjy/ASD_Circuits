import nbformat
from PIL import Image
import os

def compress_image(image_path, output_path, quality=85, max_size=(1024, 1024)):
    with Image.open(image_path) as img:
        img.thumbnail(max_size)
        img.save(output_path, quality=quality)

def process_notebook(notebook_path):
    # Load the notebook
    with open(notebook_path, 'r') as f:
        notebook = nbformat.read(f, as_version=4)
    
    # Iterate through the notebook's cells
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            # Check if the code cell contains figure generation logic and save images
            # If image files are generated and saved, you could apply compression here
            pass
        elif cell.cell_type == 'markdown':
            if '![image]' in cell.source:
                # Extract image paths (you could use regex for more robust extraction)
                # Assume images are stored in the same directory as notebook
                image_paths = [line.split('(')[-1].split(')')[0] for line in cell.source.split('\n') if '.png' in line or '.jpg' in line]
                for image_path in image_paths:
                    if os.path.exists(image_path):
                        output_path = 'compressed_' + image_path
                        compress_image(image_path, output_path)
                        # Update the notebook to use the compressed image
                        cell.source = cell.source.replace(image_path, output_path)
    
    # Save the updated notebook
    with open(notebook_path, 'w') as f:
        nbformat.write(notebook, f)

# Process all notebooks in a folder
notebooks_folder = './notebooks_mouse_str/'
for root, dirs, files in os.walk(notebooks_folder):
    for file in files:
        if file.endswith('.ipynb'):
            notebook_path = os.path.join(root, file)
            process_notebook(notebook_path)

notebooks_folder = './notebooks_mouse_sc/'
for root, dirs, files in os.walk(notebooks_folder):
    for file in files:
        if file.endswith('.ipynb'):
            notebook_path = os.path.join(root, file)
            process_notebook(notebook_path)
