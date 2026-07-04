from .func_inference import *
from .func import *
from .func_ml import *
from .impl import *
from .Models import *
import sklearn.model_selection

def extract_matching_orbits(scatterometer_data_path: str,
                            date : str ,
                            lat_range : tuple = [-90, 90],
                            lon_range : tuple = [-180, 180],
                            goes_aws_url_folder : str = "noaa-goes16/ABI-L2-CMIPF",
                            goes_channel : str = "C01",
                            goes_image_size : int = 128,
                            verbose : bool = True,
                            save : bool =True
                            ):
    
    """    Extracts matching orbits from scatterometer data (pre-downloaded) and GOES images (automatically extracted) for a given date. 
    The function filters the data to only include daylight observations and returns the images and numerical data in a dictionary format.

    Args:
        scatterometer_data_path (str): Path to the pre-downloaded scatterometer data. Data must be in .netCDF format. See tutorial for downloading data.
        date (str): Date for which to extract the data in 'YYYY-MM-DD' format.
        lat_range (tuple): Latitude range to filter the data, default is [-90,180] (all latitudes).
        lon_range (tuple): Longitude range to filter the data, default is [-180, 180] (all longitudes).
        goes_aws_url_folder (str): AWS URL folder for the GOES data, default is "noaa-goes16/ABI-L2-CMIPF".
        goes_channel (str): GOES channel to extract, default is "C01" (visible).
        goes_image_size (int): Size of the GOES images to extract, default is 128.
        verbose (bool): If True, prints additional information during the extraction process, default is True.
        save (bool): If True, saves the preloaded data to a compressed .npz file, default is True.

    Returns:
        images (list): List of filtered GOES images.
        numerical_data (dict): Dictionary containing filtered numerical data (latitudes, longitudes, times, wind speeds).
        saved_file_path (str): Path to the saved .npz file if save is True, otherwise None.

    """
    if verbose:
        print("START : Extracting matching orbits for date:", date) 
        print("INFO : lat_range:", lat_range,) 
        print("INFO : lon_range:", lon_range,) 
        print("INFO : goes_channel:", goes_channel, 'at url:', goes_aws_url_folder)

    # Extract scatterometer data in correct format

    (
        observation_times_local,
        observation_lats_local,
        observation_lons_local,
        observation_wind_speeds_local,
    ) = extract_scatter_multisat(
        scatterometer_data_path, date, lat_range, lon_range,verbose=verbose)
    
    
    # filter nighttime data
    valid_times, valid_lats, valid_lons, valid_wind_speeds = filter_nighttime(
        observation_times_local,
        observation_lats_local,
        observation_lons_local,
        observation_wind_speeds_local,
        verbose=verbose
        )
    
    # Extract GOES data from matching orbits 
    # If this is first run with dataset, it will create a folder with indices which takes time.
    
    images = extract_goes(  
        observation_times=valid_times, 
        observation_lats=valid_lats,
        observation_lons=valid_lons,
        scatterometer_data_path=scatterometer_data_path,
        goes_aws_url_folder=goes_aws_url_folder,
        goes_channel=goes_channel,
        goes_image_size=goes_image_size,
        verbose=verbose
        )

    # Package the data into a dictionary for easy access and logging. The observation_wind_speeds is the target variable.
    numerical_data = {
        "observation_lats": np.array(valid_lats),
        "observation_lons": np.array(valid_lons),
        "observation_times": np.array(valid_times),
        "observation_wind_speeds": np.array(valid_wind_speeds),
    }

    # Filter the images and numerical data to remove nans, invalid images

    images_filtered, numerical_data_filtered = package_data(
        images, numerical_data, solar_conversion=False, verbose=verbose
    )

    if verbose:
        print("END : Extracted :", len(images_filtered)," training pairs")

    # Save the preloaded data to a compressed .npz file for later use
    if save:
        
        # check if folder exists
        if not os.path.exists("./saved_files/"):
            os.makedirs("./saved_files/")

        channels_safe_name = goes_channel
        saved_file_path = f"./saved_files/file_preloaded_{date}_{channels_safe_name}.npz"
        print("INFO : Saving preloaded data to file : ", saved_file_path)

        np.savez_compressed(
            saved_file_path,
            images=images_filtered,
            numerical_data=numerical_data_filtered,
        )

        return images_filtered, numerical_data_filtered, saved_file_path


    return images_filtered, numerical_data_filtered


def train_test_model(
    data_split: dict,
    run_name: str,
    model_parameters: dict,
    normalization_factors: dict,
    ):

    """
    Trains and tests a model using the provided parameters and an already-split dataset.

    Data preparation is a separate step from training: build `data_split` with
    `prepare_data_split(saved_file_path, split_config)` in its own cell first (optionally
    inspecting it with `plot_data_split_map`), then pass the result here. This makes it
    possible to verify a spatial holdout (or any split) before spending time on training.

    Args:
        data_split (dict): Output of `prepare_data_split` (or `random_train_test_split` /
            `spatial_train_test_split` directly), with keys "train", "val", "test", each
            a dict containing "images" and "targets".
        run_name (str): Name of the run, used to create a folder for saving results.
        model_parameters (dict): Dictionary containing model parameters such as batch size, image size, learning rate, etc (Supports CNN, ResNet, ViT). See tutorial for details.
        normalization_factors (dict): Dictionary containing normalization factors (mean and std) for the images dataset.

    Returns:
        result_path_folder (str): Path to the folder where results are saved.


    model_parameters should contain the following (dictionary, values can be changed as needed):
            "batch_size" : 256,
            "image_size": 128,
            "image_channels" : 1,
            "model_choice" : "ResNet", # or "CNN" or"ViT"
            "criterion" : nn.MSELoss(), # or any other PyTorch loss function
            "optimizer_choice" : "Adam",
            "learning_rate" : 0.003305753102490767,
            "weight_decay" : 0.00000148842072509874,
            "dropout_rate" : 0.2752124679248082,
            "num_epochs" : 10,
            "patience_epochs" : 20, # early stopping
            "patience_loss" : 0.001,

            # The following additional parameters are required with the CNN :
            "activation_cnn" : nn.ReLU(),
            "activation_final" : nn.Identity(),
            "kernel_size" : 3,
            "features_cnn" : [64,64,64,64],
            "stride" : 1,

    normalization_factors should contain the following (dictionary, values can be changed as needed):
            "mean" : 0.0, # mean of the images dataset
            "std" : 1.0, # std of the images dataset

    """

    # Ignore warnings for division by zero and invalid operations
    np.seterr(divide='ignore', invalid='ignore')

    # create a folder for the experiment
    result_path_folder = create_folder(run_name)

    # load the model parameters
    batch_size = model_parameters["batch_size"]
    image_height = model_parameters["image_size"]
    image_width = model_parameters["image_size"]
    in_channels = model_parameters["image_channels"]
    lr = model_parameters["learning_rate"]
    weight_decay = model_parameters["weight_decay"]
    criterion = model_parameters["criterion"]
    optimizer_choice = model_parameters["optimizer_choice"]
    dropout_rate = model_parameters["dropout_rate"]


    num_epochs = model_parameters["num_epochs"]
    patience_epochs = model_parameters["patience_epochs"]
    patience_loss = model_parameters["patience_loss"]
    model_choice = model_parameters["model_choice"]


    # for mac devices 

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("using Apple M-series gpu")
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"INFO : Pytorch is using device: {device}")
    
    

    # Load the model

    if model_choice == "CNN":
    
        activation_cnn = model_parameters["activation_cnn"]
        activation_final = model_parameters["activation_final"]
        kernel_size = model_parameters["kernel_size"]
        features_cnn = model_parameters["features_cnn"]
        stride = model_parameters["stride"]

        model = ConventionalCNN(
            image_height,
            image_width,
            features_cnn,
            kernel_size,
            in_channels,
            activation_cnn,
            activation_final,
            stride,
            dropout_rate
        ).to(device)
        print("INFO : model choice is CNN")

    if model_choice == 'ViT':
        model = ViT(
            img_size = (128, 128),
            patch_size = (16,16),
            n_channels = 1,
            d_model = 1024,
            nhead = 4,
            dim_feedforward = 2048,
            blocks = 8,
            mlp_head_units = [1024, 512],
            n_classes = 1,
        ).to(device)
        print('INFO : model choice is ViT')

    if model_choice == 'ResNet':
        print('INFO : model choice is ResNet')
        model = ResNet50(num_classes=1, channels=1).to(device)

        # use the best saved model for test
        # print('using preloaded model, continuing training')
        # model_file = [file for file in os.listdir(path_folder) if file.endswith(".pth")][0]
        # model_path = os.path.join(path_folder, model_file)
        # model.load_state_dict(torch.load(model_path,map_location=torch.device(device)) )


    # loading normalization factors
    mean = normalization_factors["mean"]
    std = normalization_factors["std"]

    print(
        f"INFO : Using pre-split data ({data_split.get('strategy', 'unknown')} strategy) - "
        f"{len(data_split['train']['targets'])} train / {len(data_split['val']['targets'])} val / "
        f"{len(data_split['test']['targets'])} test"
    )

    # save test point locations alongside results, useful to verify/plot the held-out region later
    if "lats" in data_split["test"]:
        np.save(os.path.join(result_path_folder, "test_lats.npy"), data_split["test"]["lats"])
        np.save(os.path.join(result_path_folder, "test_lons.npy"), data_split["test"]["lons"])

    train_dataset = conventional_dataset(
        data_split["train"]["images"],
        data_split["train"]["targets"],
        #transform=Normalize(mean,std),
    )

    validation_dataset = conventional_dataset(
        data_split["val"]["images"],
        data_split["val"]["targets"],
        #transform=Normalize(mean,std),
    )

    test_dataset = conventional_dataset(
        data_split["test"]["images"],
        data_split["test"]["targets"],
        #transform=Normalize(mean,std),
    )

    # Native pytorch dataloader

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last= False,
        pin_memory=True,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last= False,
        pin_memory=True,

    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last= False,
        pin_memory=True,

    )

    # Train the model ! 

    best_val_outputs, best_val_labels, model, train_losses, val_losses = train_model(
        model,
        train_loader,
        validation_loader,
        num_epochs,
        lr,
        weight_decay,
        criterion,
        device,
        optimizer_choice,
        patience_epochs,
        patience_loss,
        result_path_folder,
    )

    # Ploting and saving the results

    plot_save_loss(
        best_val_outputs,
        best_val_labels,
        train_losses,
        val_losses,
        result_path_folder,
        saving=False,
    )

    model_file = [file for file in os.listdir(result_path_folder) if file.endswith(".pth")][0]
    model_path = os.path.join(result_path_folder, model_file)
    model.load_state_dict(torch.load(model_path,map_location=torch.device(device)) )

    # Running the test
    test_output, test_target, test_loss = test_model(model, test_loader, criterion, device)

    # Plotting the results
    error_plot(test_output, test_target, result_path_folder)
    rmse_per_range(test_output, test_target, result_path_folder)  # ADD TO METADATA

    np.save(os.path.join(result_path_folder, "test_loss.npy"),
            test_loss)


    # saving the test output and labels
    np.save(
        os.path.join(result_path_folder, "test_output.npy"),
        test_output,
    )
    np.save(
        os.path.join(result_path_folder, "test_labels.npy"),
        test_target,
    )

    return result_path_folder


def inference_full_goes_image(
    datetime: str,
    scatterometer_data_path : str,
    result_path_folder : str,
    model_parameters : dict,
    buoy_path : str,
    normalization_factors : dict,
    goes_aws_url_folder : str = "noaa-goes16/ABI-L2-CMIPF",
    goes_channel : str = 'C01',
):
    
    """ Runs the inference on a full GOES image for a given datetime using the pre-trained model from the function `train_test_model`.

    Args:
            datetime (str): Date and time in 'YYYY-MM-DD HH:MM:SS'
            scatterometer_data_path (str): Path to the pre-downloaded scatterometer data. Must be the same used in for extracting the matching orbits.
            result_path_folder (str): Path to the folder where the results of the model training are
            model_parameters (dict): Dictionary containing model parameters such as batch size, image size, learning rate, etc. Must be identical as those used in `train_test_model`.
            buoy_path (str): Path to the buoy data folder. Must be a folder containing the buoy data in `.nc` format. More details about how to download the buoy data can be found in the tutorial.
            normalization_factors (dict): Dictionary containing normalization factors (mean and std) for the images dataset.
            goes_aws_url_folder (str): AWS URL folder for the GOES data, default is "noaa-goes16/ABI-L2-CMIPF". Must be the same as used in the matching orbits extraction.
            goes_channel (str): GOES channel to extract, default is "C01" (visible).
            
    Returns:
            None, but saves the inference results in a folder named `inference_<datetime>` in the `result_path_folder`.
    """
    
    # Getting polar data from the scatterometer data path
    for file in os.listdir(scatterometer_data_path):
        if file.endswith(".nc"):
            polar_data = xr.open_dataset(
                os.path.join(scatterometer_data_path, file),
                engine="h5netcdf",
                drop_variables=["DQF"],
            )
            break
    else:
        print('WARNING : No .nc file found in the scatterometer data path, please check the path')

    if goes_channel == 'C01' or goes_channel == 'C03' or goes_channel == 'C05':
        for file in os.listdir('./satellite_indices/'):
                if file.endswith('.npy') and "1km" in file:
                    parallel_index = np.load(
                    os.path.join('./satellite_indices/', file), allow_pickle=True
            )
                    break
    elif goes_channel == 'C02':
        for file in os.listdir('./satellite_indices/'):
                if file.endswith('.npy') and "0.5km" in file:
                    parallel_index = np.load(
                    os.path.join('./satellite_indices/', file), allow_pickle=True
            )
                    break
    else :
        parallel_index = np.load('./satellite_indices/2km at nadir_index.npy', allow_pickle=True)
    
    datetime = pd.to_datetime(datetime)


    images, valid_lats, valid_lons, _ = extract_goes_production(datetime,polar_data,parallel_index,goes_channel,goes_aws_url_folder)
    lat_inference, lon_inference, wind_speeds_inference = inference_whole_image(result_path_folder,images,valid_lats,valid_lons,model_parameters,normalization_factors,
    )

    buoy_lat,buoy_lon,buoy_time,buoy_wind_speed,buoy_name = buoy_data_extract(buoy_path,polar_data,datetime)

    safe_file_name = datetime.strftime('%Y-%m-%d_%H-%M-%S')
    path_inference = f'{result_path_folder}/inference_{safe_file_name}/'
    if not os.path.exists(path_inference):
        os.makedirs(path_inference)
    
    print('INFO : Folder of inference created:', path_inference)

    goes_image = plot_goes_image(lat_inference,lon_inference,images,path_inference,buoy_name,buoy_lat,buoy_lon,datetime)
    plot_wind_speeds(lat_inference,lon_inference,wind_speeds_inference,path_inference,buoy_name,buoy_lat,buoy_lon,datetime)

    np.save(f'{path_inference}/data_goes_image.npy',goes_image)
    np.save(f'{path_inference}/data_wind_speeds.npy',wind_speeds_inference)
    np.save(f'{path_inference}/data_lat.npy',lat_inference)
    np.save(f'{path_inference}/data_lon.npy',lon_inference)
    
    reshaped_images = np.reshape(images, (160,340,128,128))

    for i in range(len(buoy_wind_speed)):
        try:
            lat_index = np.where(lat_inference == buoy_lat[i])[0][0]
            lon_index = np.where(lon_inference == buoy_lon[i])[1][0]

            buoy_image =  reshaped_images[lat_index,lon_index]
            threshold = np.max(buoy_image) * 0.6
            cloud_mask = np.where(buoy_image > threshold, 1,0)
            percentage_cloud = np.sum(cloud_mask)/np.size(cloud_mask)

            wind_speeds_inference_buoy = wind_speeds_inference[lat_index,lon_index]
            difference = wind_speeds_inference_buoy - buoy_wind_speed[i]
            print(f'EVAL : Buoy {buoy_name[i]} - Inference Wind Speed : {wind_speeds_inference_buoy} - Buoy Wind Speed : {buoy_wind_speed[i]} - Difference : {difference}, Percentage Cloud : {percentage_cloud}')

            update_buoy_comparison_csv(
                result_path_folder, datetime, buoy_name, i,
                wind_speeds_inference_buoy, buoy_wind_speed, difference, percentage_cloud
            )

        except:
            print(f'EVAL : Buoy {buoy_name[i]} not in the inference area')
            continue