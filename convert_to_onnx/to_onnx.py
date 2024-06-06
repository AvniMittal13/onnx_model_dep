from model import M3DNCA
import onnx
import torch

# only channel_n and input_size parameters are used from the config_data
# same as original config_data used for training

config_data = [{
    'img_path': r"/scratch/avinim.scee.iitmandi/datasets/Task03_Liver_rs/imagesTr",
    'label_path': r"/scratch/avinim.scee.iitmandi/datasets/Task03_Liver_rs/labelsTr",
    'name': r'M3D_NCA_liver1_256', #12 or 13, 54 opt,
    'device':"cuda",
    'unlock_CPU': True,
    # Optimizer
    'lr': 16e-4,
    'lr_gamma': 0.9999,#0.9999,
    'betas': (0.9, 0.99),
    # Training
    'save_interval': 20,
    'evaluate_interval': 20,
    'n_epoch': 3000,
    'batch_duplication': 1,
    # Model
    'channel_n': 16,        # Number of CA state channels       ## USED
    'inference_steps': [10, 10, 10, 10],
    'cell_fire_rate': 0.5,
    'batch_size': 2,
    'input_channels': 1,
    'output_channels': 1,
    'hidden_size': 64,
    'train_model':1,
    # Data
    'input_size': [(32,32,12),(64,64,24),(128,128,48),(256, 256, 96)] ,     ## USED
    'scale_factor': 2,
    'data_split': [0.9, 0, 0.1],
    'keep_original_scale': False,
    'rescale': True,
}
]

def main():

    model_path = r"/content/rsna_3000_model.pth"

    # config_path = r"C:\Users\AVNI\Desktop\6th SEM\Neullar Cellular Automata\NCA-main\nca_try\models\liver\3_32_64_128\config.dt"
    # with open(config_path, 'r') as file:
    #     config_data = json.load(file)

    first_dict = config_data[0]
    channel_n = first_dict.get("channel_n")
    input_size = first_dict.get("input_size")[-1]
    print("Channel_n:", channel_n)
    print("Input_size:", input_size)

    ## code to convert to onnx model

    ## MODIFY if required
    pytorch_model = M3DNCA(channel_n=channel_n,fire_rate= 0.5, hidden_size=64, kernel_size=7, input_channels=config_data[0]['input_channels'], output_channels=config_data[0]['output_channels'],levels=len(config_data[0]['input_size']), scale_factor=config_data[0]['scale_factor'], steps=10)

    # pytorch_model = BasicNCA3D(16, 0.5, hidden_size = 64, kernel_size= 3 + 4*index )
    pytorch_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    pytorch_model.eval() # bypass dropout layers

    dummy_input = torch.zeros([1,input_size[0],input_size[1],input_size[2],1]) # dummmy input

    torch.onnx.export(pytorch_model, dummy_input, r'nca_model_onnx'+'.onnx', verbose = True)

if __name__ == '__main__':
    main()