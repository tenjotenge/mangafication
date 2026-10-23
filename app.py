"""Local CPU Gradio preview for the deterministic baselines."""
from pathlib import Path
import tempfile

import gradio as gr
from PIL import Image

from src.baseline import METHODS, SEMANTIC_METHOD, process, contact_sheet
from src.semantic import MODEL_PATH


def run(image, method, ink, contour, detail):
    if image is None:
        raise gr.Error('Select an image first.')
    original = Image.fromarray(image).convert('RGB')
    result, shadows, contours = process(original, method, ink, contour, detail)
    variants = {name: (result if name == method else process(original, name, ink, contour, detail)[0]) for name in METHODS}
    if MODEL_PATH.exists():
        variants[SEMANTIC_METHOD] = result if method == SEMANTIC_METHOD else process(original, SEMANTIC_METHOD, ink, contour, detail)[0]
    export = Path(tempfile.mkdtemp(prefix='mangafication-'))
    for name, output in [('binary', result), ('shadows', shadows), ('contours', contours)]:
        output.save(export / f'{name}.png')
    return result, contact_sheet(original, variants), [str(p) for p in export.glob('*.png')]


with gr.Blocks(title='Mangafication baseline') as demo:
    gr.Markdown('# Mangafication · two-tone baseline')
    with gr.Row():
        source = gr.Image(label='Local photograph', type='numpy', sources=['upload'])
        result = gr.Image(label='Strictly black and white output')
    method = gr.Dropdown(METHODS + (SEMANTIC_METHOD,), value=SEMANTIC_METHOD if MODEL_PATH.exists() else 'composite', label='Method')
    ink = gr.Slider(0, 1, value=0.45, step=0.01, label='Black ink coverage')
    contour = gr.Slider(0, 1, value=0.55, step=0.01, label='Contour strength')
    detail = gr.Slider(0, 1, value=0.55, step=0.01, label='Detail preservation')
    button = gr.Button('Process and compare')
    comparison = gr.Image(label='Baseline comparison')
    files = gr.File(label='Export binary PNG and masks', file_count='multiple')
    button.click(run, [source, method, ink, contour, detail], [result, comparison, files])

if __name__ == '__main__':
    demo.launch(server_name='127.0.0.1')
