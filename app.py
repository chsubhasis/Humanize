import gradio as gr
import os
import traceback
from typing import List
from generator.BRDGenerator import SAPBRDGenerator
from processor.DocumentProcessor import SAPDocumentProcessor

#Take the key from user. Dont hardcode.
# Initialize BRD Generator
brd_generator = SAPBRDGenerator("0Av9Aj4G9Jx4fOlGmSikkhOXmCkXImyZ")

def create_brd_interface(example_assessments: List[str] = None, example_brds: List[str] = None):
    """Create Gradio interface for BRD generation."""
    
    # Load few-shot examples if provided
    if example_assessments and example_brds:
        brd_generator.load_few_shot_examples(example_assessments, example_brds)

    # Create Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown("# Interactive Business Requirements Document (BRD) Generator")
        
        with gr.Row():
            with gr.Column():
                # Assessment Upload
                assessment_input = gr.File(label="Upload Assessment Report")
                generate_btn = gr.Button("Generate BRD")
                
                # Feedback Input
                feedback_input = gr.Textbox(label="Refine BRD (Provide Feedback)", lines=5, interactive=True)
                refine_btn = gr.Button("Refine BRD")
            
            with gr.Column():
                # BRD Content Display
                brd_output = gr.Textbox(label="BRD Content", lines=15, interactive=True)
                
                # BRD Download
                brd_download = gr.File(label="Download BRD")
        
        # Event Handlers
        generate_btn.click(
            generate_new_BRD, 
            inputs=assessment_input, 
            outputs=[brd_output, brd_download]
        )
        
        refine_btn.click(
            updated_existing_BRD, 
            inputs=feedback_input, 
            outputs=[brd_output, brd_download]
        )

    return demo

def generate_new_BRD(assessment_file):
    try:
        # Extract text from uploaded assessment file
        assessment_text = SAPDocumentProcessor.extract_text(assessment_file.name)
        #print("assessment_text ready", assessment_text)
        #assessment_text = SAPDocumentProcessor.extract_text(assessment_file)
        
        # Generate BRD
        brd_content = brd_generator.generate_brd(assessment_text)
        #print("brd_content ready", brd_content)
        
        # Save BRD
        brd_filepath = brd_generator.save_brd(brd_content)
        #print("brd content saved at", brd_filepath)
        
        return brd_content, gr.File(value=brd_filepath)
    
    except Exception as e:
        # Capture the full stack trace
        error_message = f"Error: {str(e)}\n\nStack Trace:\n{traceback.format_exc()}"
        print(error_message)
        return error_message, None

def updated_existing_BRD(feedback):
    try:
        # Refine BRD
        refined_brd = brd_generator.refine_brd(feedback)
        
        # Save refined BRD
        brd_filepath = brd_generator.save_brd(refined_brd, 'refined_brd.docx')
        
        return refined_brd, gr.File(value=brd_filepath)
    
    except Exception as e:
        # Capture the full stack trace
        error_message = f"Error: {str(e)}\n\nStack Trace:\n{traceback.format_exc()}"
        print(error_message)
        return error_message, None

if __name__ == "__main__":

    EXAMPLE_ASSESSMENTS = [
        'assessment_1.pdf',
        'assessment_2.pdf'
    ]
    EXAMPLE_BRDS = [
        'brd_1.docx',
        'brd_2.docx'
    ]

    # Create interface with optional few-shot examples
    demo = create_brd_interface(EXAMPLE_ASSESSMENTS, EXAMPLE_BRDS)
    demo.launch(debug=True)

    #generate_new_BRD("new_assessment.pdf")


