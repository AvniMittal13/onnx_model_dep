var organData = {
    liver: {
      file_name: "models\\nca_model_liver1.onnx",
      required_dims: [64,64,40],
      input_var_name : "x.1",
      output_var_name : "3654"

    },
    // rsna: {
    //   file_name: "models\\nca_model_rsna_256.onnx",
    //   required_dims: [256, 256, 96],
    //   input_var_name : "x.1",
    //   output_var_name : "3723"
    // }
    rsna: {
      file_name: "models\\nca_model_rsna_128.onnx",
      required_dims: [128, 128, 48],
      input_var_name : "x.1",
      output_var_name : "3723"
    }
  };
  
export default organData;