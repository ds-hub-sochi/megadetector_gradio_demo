""" Gradio Demo for image detection"""

import os
 
import gradio as gr
import numpy as np
import supervision as sv
import torch
from PIL import Image

from PytorchWildlife.models import classification as pw_classification
from PytorchWildlife.models import detection as pw_detection

DEVICE: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(DEVICE)

dot_annotator = sv.DotAnnotator(radius=6)
box_annotator = sv.BoxAnnotator(thickness=4)
lab_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK, text_thickness=4, text_scale=2)

os.makedirs(os.path.join("..", "temp"), exist_ok=True) # ASK: Why do we need this?
os.makedirs(os.path.join(".", "checkpoints"), exist_ok=True)

detection_model = None
classification_model = None

def single_image_detection(
    input_img: Image,
    det_conf_thres: float,
    clf_conf_thres: float,
    img_index=None,
):
    """Performs detection on a single image and returns an annotated image.

    Args:
        input_img (PIL.Image): Input image in PIL.Image format defaulted by Gradio.
        det_conf_thre (float): Confidence threshold for detection.
        clf_conf_thre (float): Confidence threshold for classification.
        img_index: Image index identifier.
    Returns:
        annotated_img (PIL.Image.Image): Annotated image with bounding box instances.
    """

    input_img = np.array(input_img)
    
    annotator = box_annotator
    results_det = detection_model.single_image_detection(
        input_img,
        img_path=img_index,
        conf_thres = det_conf_thres,
    )
    
    if classification_model is not None:
        labels = []
        for xyxy, det_id in zip(results_det["detections"].xyxy, results_det["detections"].class_id):
            # Only run classifier when detection class is animal
            if det_id == 0:
                cropped_image = sv.crop_image(image=input_img, xyxy=xyxy)
                results_clf = classification_model.single_image_classification(cropped_image)
                labels.append("{} {:.2f}".format(
                    results_clf["prediction"] if results_clf["confidence"] > clf_conf_thres else "Unknown",
                    results_clf["confidence"],
                    )
                )
            else:
                labels.append("Unknown animal")
    else:
        labels = results_det["labels"]

    annotated_img = lab_annotator.annotate(
        scene=annotator.annotate(
            scene=input_img,
            detections=results_det["detections"],
        ),
        detections=results_det["detections"],
        labels=labels,
    )

    return annotated_img


with gr.Blocks() as demo:
    classification_model = pw_classification.Stage2Classifier().to(DEVICE)
    detection_model = pw_detection.MegaDetectorV6(pretrained=True, device=DEVICE)

    gr.Markdown("# Pytorch-Wildlife Demo")

    with gr.Tab("Single Image Process"):
        with gr.Row():
            with gr.Column():
                sgl_in = gr.Image(type="pil")

                sgl_conf_sl_det = gr.Slider(
                    0,
                    1,
                    label="Detection Confidence Threshold",
                    value=0.2,
                )

                sgl_conf_sl_clf = gr.Slider(
                    0,
                    1,
                    label="Classification Confidence Threshold",
                    value=0.7,
                    visible=True,
                )

            sgl_out = gr.Image()
        sgl_but = gr.Button("Detect Animals!")

    sgl_but.click(
        single_image_detection,
        inputs=[sgl_in, sgl_conf_sl_det, sgl_conf_sl_clf],
        outputs=sgl_out,
    )

if __name__ == "__main__":
    demo.launch(share=True)
