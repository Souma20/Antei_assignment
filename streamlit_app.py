import streamlit as st
import traceback
from PIL import Image

from model import HealthGPT_Agent
from config import HealthGPTConfig_M3_COM, HealthGPTConfig_M3_GEN, HealthGPTConfig_L14_COM

configs = {
    "HealthGPT-M3-COM": HealthGPTConfig_M3_COM(),
    "HealthGPT-M3-GEN": HealthGPTConfig_M3_GEN(),
    "HealthGPT-L14-COM": HealthGPTConfig_L14_COM()
}

agent = HealthGPT_Agent(configs=configs, model_name=None)

'''--------UI--------'''

st.title("🖼️ HealthGPT Streamlit Demo")
st.write("This demo lets you test HealthGPT for medical image analysis or generation.")

option = st.radio("🔍 Choose the task", ["Analyze Image", "Generate Image"])

model_radio = st.radio("🧠 Choose the model", ["HealthGPT-M3", "HealthGPT-L14"])

text_input = st.text_area("Question", 
                          "Could you explain what this mass in the MRI means for my health? Is it very serious?")

uploaded_file = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image_input = Image.open(uploaded_file).convert("RGB")
    st.image(image_input, caption="Uploaded Image", use_column_width=True)
else:
    image_input = None

if st.button("🚀 Process"):
    if not text_input.strip():
        st.warning("⚠️ Please input your question.")
    else:
        st.info("Processing your request...")
        try:
            # Select the model based on the task
            if option == "Analyze Image":
                selected_model = model_radio + "-COM"
            elif option == "Generate Image":
                selected_model = model_radio + "-GEN"
            
            # Load the model if not already loaded
            agent.load_model(selected_model)
            
            # Process the input using the HealthGPT agent
            response = agent.process(option, text_input, image_input)
            
            # Display output based on task
            if option == "Analyze Image":
                st.markdown("### HealthGPT Answer")
                st.text_area("", value=response, height=300)
            elif option == "Generate Image":
                st.image(response, caption="Generated Image", use_column_width=True)
                
        except Exception as e:
            st.error(f"⚠️ An error occurred: {str(e)}")
            st.error(traceback.format_exc())

st.markdown("""
### Terms of Use
By using this service, you agree that this is a research preview intended for non-commercial use only. It provides limited safety measures and may generate offensive content. Do not use this service for any illegal, harmful, or unethical purposes.
""")
