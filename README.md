# ONNX Model Deployment for M3D-NCA - Medical Image Segmentation

This repository provides tools and scripts for deploying the M3D-NCA model for medical image segmentation using ONNX. The guide below includes setup instructions, steps to convert a trained PyTorch model to ONNX format, and instructions for adding new models to the web application.

## Setup

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Create and activate the environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
    
## Converting a Trained PyTorch Model to ONNX Format
### For M3D-NCA Rewritten (Currently Integrated)

1. **Modify `config_data` in `to_onnx.py`:**
   Update the `config_data` to match the configuration used during training. Only the `channel_n` and `input_size` parameters are required.
   
   ```python
   config_data = {
       "channel_n": <CHANNEL_N_VALUE>,
       "input_size": <INPUT_SIZE_VALUE>
   }
   ```
 
2. **Modify the model path name:**
    Ensure the model path in to_onnx.py points to the correct trained model file.

   ```python 
   model_path = "path/to/your/model.pth"
   ```

3. **Run the script to convert to ONNX:**

   ```sh
   python to_onnx.py
   ```
   
### For M3D-NCA Old (Saving Each NCA Model Separately)

1. **Modify `model_path` and `config_path` parameters in `convert_basicNCA_model.py`:**
   
   ```python
   model_path = "path/to/your/old_model.pth"
   config_path = "path/to/your/config.json"
   ```

2. **Run the script to convert to ONNX:**
   ```sh
   python convert_basicNCA_model.py
   ```
   
   
## Adding a New Model to the Web Application

1. **Convert the trained M3D-NCA model to ONNX format:**
   Follow the steps mentioned above to convert your model to ONNX format.

2. **Upload the ONNX model to Netron:**
   Go to [Netron](https://netron.app/) and upload your ONNX model. Check the input and output variable names.

3. **Modify `organData.js`:**
   Update `organData.js` with the following information:
   - Path to the ONNX model file
   - Input dimensions used for creating the ONNX file
   - Input and output variable names obtained from Netron

   Example:
   ```javascript
   var organData = {
       "organ_name": {
           "file_name": "path/to/your/model.onnx",
           "required_dims": [batch_size, channels, height, width],
           "input_var_name": "input_variable_name",
           "output_var_name": "output_variable_name"
       }
   }
   ```
   
## Repository Structure

- `requirements.txt` - List of dependencies
- `to_onnx.py` - Script for converting M3D-NCA rewritten models to ONNX
- `convert_basicNCA_model.py` - Script for converting older M3D-NCA models to ONNX
- `organData.js` - Configuration file for web application

---

By following these instructions, you should be able to successfully convert and deploy M3D-NCA models for medical image segmentation. If you encounter any issues, please refer to the provided scripts and configurations or reach out for support.

