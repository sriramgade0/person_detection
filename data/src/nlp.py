from transformers import pipeline

# Load a captioning model
captioner = pipeline('image-to-text')

def describe_detection(frame):
    description = captioner(frame)
    return description[0]['generated_text']
