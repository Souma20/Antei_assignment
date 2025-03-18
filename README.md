## Tasks

### Task 1: Explore & Launch HealthGPT Locally
**Objective:**  
Explore, test, and prototype HealthGPT by running it locally, evaluating its performance, and enhancing its functionality.

**Completed:**
- **Repository & Environment Setup:**
  - Cloned the repository.
  - Configured the Python environment and installed all dependencies.
- **Local Execution:**
  - Successfully ran HealthGPT on the local machine using both the Gradio interface and a custom Streamlit demo.
- **Architecture & Capability Review:**
  - **Hierarchical Visual Perception:**  
    Processes medical images at multiple scales to capture overall structures (e.g., organ shapes) and fine details (e.g., small lesions).
  - **H-LoRA (Heterogeneous Low-Rank Adaptation):**  
    Uses specialized, parameter-efficient adapter layers to support multi-modal tasks without retraining the entire model.
  - **Task-Specific Hard Router:**  
    Dynamically selects the appropriate modules for either visual comprehension or image generation.
  - **Autoregressive Outputs:**  
    Generates text responses token-by-token for comprehension and image tokens for generation.

---

### Task 2: Test on Multiple Datasets
**Objective:**  
Evaluate how well HealthGPT processes and analyzes diverse medical images.

**Completed:**
- **Functional Testing:**
  - Tested the model on multiple images downloaded from online sources.
  - Verified that HealthGPT produces correct outputs for different types of images.
- **Observations:**
  - The model processes images correctly.
  - Inference times are long.
  - The system is resource-intensive and may crash on low-end computers.

---

### Task 3: Build a Streamlit Demo
**Objective:**  
Create a user-friendly interface that replicates the core functionalities of the Gradio interface using Streamlit.
![Screenshot 2025-03-17 224457](https://github.com/user-attachments/assets/159f43e3-f623-48cf-bb6a-aca11dbc2efd)

- **Interface Development:**
  - Recreated input boxes for text (to enter questions) and image uploads.
  - Added options to select between analysis (comprehension) and generation tasks.
- **Integration:**
  - Successfully integrated the HealthGPT agent into the Streamlit app.
  - Enabled interactive testing through a user-friendly interface.

---

### Task 4: Write an Analytical Report
**Objective:**  
Document the strengths, weaknesses, and potential applications of HealthGPT.

**Completed:**
- **Advantages Identified:**
  - Designed for multi-modal processing with a unified architecture.
  - Supports simultaneous text and image generation.
  - Efficiently adapts to new tasks through H-LoRA.
- **Disadvantages Identified:**
  - Testing was performed on a limited set of images.
  - Long inference times and high resource usage may lead to crashes on low-end PCs.
  - Encountered challenges with image preprocessing.
- **Potential Applications:**
  - Clinical radiology assistance.
  - Medical education and research.
  - Telemedicine for remote diagnosis.
- **Outcome:**
  - Recorded detailed observations and insights based on the testing phase.

---

### Task 5: Suggest & Implement Enhancements
**Objective:**  
Propose and, if possible, implement improvements to enhance the overall performance and usability of HealthGPT.

**Completed:**
- **Proposed Improvements:**
  - Enhance image preprocessing techniques (e.g., resizing and normalization) to ensure consistent input quality.
  - Optimize performance using methods such as quantization or model distillation to reduce inference time.
  - Upgrade the user interface with additional features like confidence scoring and batch processing.
  - **Add an Emergency Contact Feature:**
    - Provide an option for users to input emergency contact details.
    - If the model detects a critical or urgent condition based on its analysis, notify the listed contact.
    - Integrate with email or SMS APIs to send real-time alerts.


## Installation

### 1. Prepare Environment

First, clone our repository and create the Python environment for running HealthGPT using the following command:

```bash
# clone our project
git clone https://github.com/DCDmllm/HealthGPT.git
cd HealthGPT

# prepare python environment
conda create -n HealthGPT python=3.10
conda activate HealthGPT
pip install -r requirements.txt
```

### 2. Prepare Pre-trained Weights

HealthGPT utilizes clip-vit-large-patch14-336 as the visual encoder and employs Phi-3-mini-4k-instruct and phi-4 as the pre-trained LLM base models for HealthGPT-M3 and HealthGPT-L14, respectively. Please download the corresponding weights:

| Model Type | Model Name | Download |
|------------|------------|----------|
| ViT | clip-vit-large-patch14-336 | [Download](https://github.com/DCDmllm/HealthGPT) |
| Base Model (HealthGPT-M3) | Phi-3-mini-4k-instruct | [Download](https://github.com/DCDmllm/HealthGPT) |
| Base Model (HealthGPT-L14) | phi-4 | [Download](https://github.com/DCDmllm/HealthGPT) |

For medical vision generation tasks, please follow the official VQGAN guide and download the VQGAN OpenImages (f=8), 8192 model weights from the "Overview of pretrained models" section. Below is the direct link to the corresponding VQGAN pre-trained weights:

| Model Name | Download |
|------------|----------|
| VQGAN OpenImages (f=8), 8192, GumbelQuantization | [Download](https://github.com/DCDmllm/HealthGPT) |

After downloading, place the `last.ckpt` and `model.yaml` files in the `taming_transformers/ckpt` directory.

### 3. Prepare H-LoRA and Adapter Weights

HealthGPT enhances the base model's capabilities for medical visual comprehension and generation by training a small number of H-LoRA parameters and adapter layers for aligning vision and text. We have currently released some weights from the training process, supporting medical visual question answering and open-world visual reconstruction tasks. Here are the corresponding weights: [Download](https://github.com/DCDmllm/HealthGPT).

We will soon be releasing the full weights for HealthGPT-L14, along with the H-LoRA weights for medical generation tasks. Stay tuned!!!

## ⚡ Inference

### Medical Visual Question Answering

To perform inference using HealthGPT, please follow these steps:

1. **Download Necessary Files**:
   - Ensure you have downloaded all the required model weights and resources.

2. **Update Script Paths**:
   - Open the script located at `llava/demo/com_infer.sh`.
   - Modify the following variables to point to the paths where you stored the downloaded files:
     - `MODEL_NAME_OR_PATH`: Path or identifier for base model.
     - `VIT_PATH`: Path to the Vision Transformer model weights.
     - `HLORA_PATH`: Path to the HLORA weights file for visual comprehension.
     - `FUSION_LAYER_PATH`: Path to your fusion layer weights file.

3. **Run the Script**:
   - Execute the script in your terminal to begin inference:
   ```bash
   cd llava/demo
   bash com_infer.sh
   ```

You can directly run the Python command in your terminal by specifying the paths and parameters. This approach allows you to easily change the image or question as needed:

```bash
python3 com_infer.py \
    --model_name_or_path "microsoft/Phi-3-mini-4k-instruct" \
    --dtype "FP16" \
    --hlora_r "64" \
    --hlora_alpha "128" \
    --hlora_nums "4" \
    --vq_idx_nums "8192" \
    --instruct_template "phi3_instruct" \
    --vit_path "openai/clip-vit-large-patch14-336/" \
    --hlora_path "path/to/your/local/com_hlora_weights.bin" \
    --fusion_layer_path "path/to/your/local/fusion_layer_weights.bin" \
    --question "Your question" \
    --img_path "path/to/image.jpg"
```

Customize the Question and Image: You can modify the `--question` and `--img_path` parameters to ask different questions or analyze different images.

Correspondingly, the visual Question Answering task of HealthGPT-L14 can be executed with the following Python command:

```bash
python3 com_infer_phi4.py \
    --model_name_or_path "microsoft/Phi-4" \
    --dtype "FP16" \
    --hlora_r "32" \
    --hlora_alpha "64" \
    --hlora_nums "4" \
    --vq_idx_nums "8192" \
    --instruct_template "phi4_instruct" \
    --vit_path "openai/clip-vit-large-patch14-336/" \
    --hlora_path "path/to/your/local/com_hlora_weights_phi4.bin" \
    --question "Your question" \
    --img_path "path/to/image.jpg"
```

The weights of `com_hlora_weights_phi4.bin` can be downloaded [here](https://github.com/DCDmllm/HealthGPT).

### Image Reconstruction

Similarly, simply set the `HLORA_PATH` to point to the `gen_hlora_weights.bin` file and configure the other model paths. Then, you can perform the image reconstruction task using the following script:

```bash
cd llava/demo
bash gen_infer.sh
```

You can also directly execute the following python command:

```bash
python3 gen_infer.py \
    --model_name_or_path "microsoft/Phi-3-mini-4k-instruct" \
    --dtype "FP16" \
    --hlora_r "256" \
    --hlora_alpha "512" \
    --hlora_nums "4" \
    --vq_idx_nums "8192" \
    --instruct_template "phi3_instruct" \
    --vit_path "openai/clip-vit-large-patch14-336/" \
    --hlora_path "path/to/your/local/gen_hlora_weights.bin" \
    --fusion_layer_path "path/to/your/local/fusion_layer_weights.bin" \
    --question "Reconstruct the image." \
    --img_path "path/to/image.jpg" \
    --save_path "path/to/save.jpg"
```

## Server

An interactive Chat UI based on Gradio, supporting text + image input, and returning text or images according to different modes.

### 📌 Project Introduction

This project is a Gradio front-end interface, supporting users:

- Analyze image (comprehension task): input text + image, output text
- Generate image (generation task): input text + image, output image

### 📦 Installation Dependencies

This project runs based on Python, and requires the installation of Gradio, Pillow, and Streamlit.

```bash
pip install gradio pillow streamlit
```

### ▶️ Run the project

Run the following command in the terminal:

```bash
python app.py
```

To run the Streamlit interface:

```bash
streamlit run streamlit_app.py
```
