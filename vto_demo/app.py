import gradio as gr
import yaml
import argparse
from pipeline_loader import PipelineLoader
from front_end import append_gradio_frontend
from request_processor import TryOnProcessor
from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser(description="Run the Gradio app with specified configurations.")
parser.add_argument('--config_path', type=str, default='../configs/pipeline_config.yaml', help='Path to the pipeline configuration YAML file.')
parser.add_argument('--port', type=int, default=7834, help='Port number for the Gradio server.')
parser.add_argument('--host', type=int, default="0.0.0.0", help='Host for the Gradio server.')


def main(args, pipeline_config):
    pipeline_loader = PipelineLoader(
        base_path=pipeline_config['base_path'],
    )

    try_on_processor = TryOnProcessor(
        pipeline_config, pipeline_loader
    )

    image_blocks = gr.Blocks().queue()
    with image_blocks as demo:
        append_gradio_frontend(
            example_path = pipeline_config['example_path'],
            processing_fn=try_on_processor.start_tryon
        )

    image_blocks.launch(server_port=args.port, server_name=args.host)

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config_path, 'r') as file:
        pipeline_config = yaml.safe_load(file)
    main(pipeline_config)
