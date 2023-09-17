# Wizard_QLoRA_Finetuning
Finetuning Some Wizard Models With QLoRA

# Demo
[https://youtu.be/hkt5Nz0buso?si=HNmYLp_z5SGZlMbM](https://youtu.be/hkt5Nz0buso?si=HNmYLp_z5SGZlMbM)

# Pipeline
## Model fine-tuning
Finetuning can be done with the `finetune.py` script. In this script, a model will be downloaded and finetuned on one of the datasets in 4-bit precision. 
As finetuning progress is being made, checkpoints are saved to the specified output directory.

## Merging
After the model is trained, one of the checkpoint files should be merged so that the LoRA weights and old weights are combined into a single weight matrix, 
making inference more efficient than if you had them split. `merge.py` does the merge given a specified checkpoint file and the specified model type.

## Inference
Inference has a few scripts. `infer.py` and `infer.ipynb` are similar and just run straight inference on a given model. 
`infer_interface.ipynb` has an additional interface using Gradio.

## Uploading/Saving Models
`upload.py` can be used to upload huggingface models to the hub easily given a repo name to upload. Make sure to get a `write` token from huggingface to upload properly.

## Data Creation
`data_creation.ipynb` is a simple example of data creation.
