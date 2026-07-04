import windscangeo
import datetime
from torch import nn

images, numerical_data, saved_file_path = windscangeo.extract_matching_orbits(scatterometer_data_path= "./Copernicus_Scatterometer/",
                                    date= '2022-03-01',
                                    lat_range= [-90, 90],
                                    lon_range=[-180, 180],
                                    goes_aws_url_folder= "noaa-goes16/ABI-L2-CMIPF", #or "noaa-goes16/ABI-L1b-RadF" (rest not yet optimized, at your own risk !) https://registry.opendata.aws/noaa-goes/
                                    goes_channel= 'C06',
                                    goes_image_size = 128,
                                    verbose= True,
                                    save = True,
)

split_config_spatial = {
    "strategy": "spatial",
    "test_region": {"lat_min": 0, "lat_max": 10, "lon_min": -40, "lon_max": -20},  # region held out entirely for testing
    "goes_channel": "C06",  # must match goes_channel used in extract_matching_orbits, used to size the buffer
    "buffer_deg": None,     # None = derive from patch footprint; widen if domain is far off-nadir
    "val_fraction": 0.1,
}

data_split_spatial = windscangeo.prepare_data_split(saved_file_path, split_config_spatial)
fig_spatial = windscangeo.plot_data_split_map(data_split_spatial)


model_parameters = {
    "batch_size" : 256,
    "image_size": 128, 
    "image_channels" : 1,  
    "model_choice" : "ResNet", # or "CNN" or"ViT"
    "criterion" : nn.MSELoss(), # or any other PyTorch loss function
    "optimizer_choice" : "Adam", 
    "learning_rate" : 0.003305753102490767,
    "weight_decay" : 0.00000148842072509874,
    "dropout_rate" : 0.2752124679248082,
    "num_epochs" : 1, 
    "patience_epochs" : 20, # early stopping
    "patience_loss" : 0.001,

    
}

normalization_factors = { # Normalization factors for the input data, 
    "mean" : 0,
    "std" : 1
}


# spacial data split 

result_path_folder = windscangeo.train_test_model(data_split_spatial,
                                            run_name= 'test_run_spatial',
                                            model_parameters= model_parameters,
                                            normalization_factors= normalization_factors,
                                            )

hours = [10,11,12,13,14,15,16,17,18]

for hour in hours:
    specific_time = datetime.datetime(2022, 3, 1, hour, 0, 0) # specify the time you want to run the inference for

    windscangeo.inference_full_goes_image(datetime = specific_time, # run a loop with different times 
                                        scatterometer_data_path= "./Copernicus_Scatterometer/",
                                        result_path_folder = result_path_folder,
                                        model_parameters = model_parameters,
                                        buoy_path= "./buoy_validation/data/",
                                        normalization_factors = normalization_factors,
                                        goes_aws_url_folder= "noaa-goes16/ABI-L2-CMIPF", # must be the same as used in training
                                        goes_channel= 'C06', # must be the same as used in training
    )
    