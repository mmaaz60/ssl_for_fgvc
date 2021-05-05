# Configuration

The table below list the details of the configurable parameters in [config.yml](config.yml).

| Name| Parameter      | Description | Expected Value    |
| :---        |:---  | :--- | :--- |
|general|||
| |output_directory|Name of main directory to save results and checkpoints| string: name of directory|
| |experiment_id|Name for result folder and log file| string: name of experiment|
| |model_checkpoints_directory_name|Name of directory to save checkpoints| string: name for checkpoint directory|
|dataloader| | |
| |name|Name of the dataloader to be used| string: cub_200_2011, cub_200_2011_contrastive, dcl||
| |train_data_fraction|Fraction of the training data to be used| float: Any value in the range [0,1]|
| |test_data_fraction|Fraction of the training data to be used| float: Any value in the range [0,1]|
| |download|Flag to indicate if dataset must be download|bool: True, False|
| |root_directory_path|The root directory path where dataset is downloaded|string: path to directory ('./data/CUB_200_2011' if download=True)|
| |resize_width|Image resize width before applying transforms, if any|int: any integer value. eg. 600, 1200, 2400
| |batch_size|Image resize width before applying transforms, if any|int: any integer value. eg. 600, 1200, 2400
| |shuffle|Flag to indicate if dataset for training be shuffled|bool: True, False|
| |num_workers|Number of parallel workers to load the dataset|int: any integer value, e.g. 4, 8, 16|
| |transforms:common|Set of transforms applied commonly to baseline and SSl models|
| |transforms:jigsaw:size|Size of image patch in DCL|int: [7,7] (not configurable to other dimensions)|
| |transforms:jigsaw:swap_range|Range of distance a patch can move in jigsaw shuffle|int: any integer in range of number of patches eg. 2,3|
|model| | |
| |name|Name or source of model |string: torchvision, fgvc_resnet, torchvision_ssl_rotation, fgvc_ssl_rotation, torchvision_ssl_pirl, dcl|
| |model_function_path|Path of backbone model class|string: torchvision.models.resnet50|
| |pretrained|Flag to indicate if weights from pretrained imagenet model must be used|bool: True, False|
| |classes_count|Number of classes for the classification problem|int: Number of classes, eg. 200 for CUB data
| |prediction_type|Type of prediction head for jigsaw locations in DCL|string: regression, class
| |checkpoints_path|Path of pre-trained weights for entire model, if required|string: Path to pre-trained weights
|diversification_block| | |
| |p_peak |Probability for peak suppression|float: Any value in the range [0,1] eg. 0.5
| |p_patch |Probability for patch suppression|float: Any value in the range [0,1] eg. 0.5
| |patch_size |Size of patch in patch suppression|int: any integer value, e.g. 3
| |alpha |Overall suppression factor controls degree of suppression|float: Any value in the range [0,1] eg. 0.1
| |use_during_test |Flag to indicate if diversification block should be used during test|bool: True, False|
|train| | |
| |name|Name of the trainer to be used|string: base_trainer, ssl_rot_trainer, ssl_pirl_trainer, dcl_trainer|
| |epochs|Number of epochs|int: any integer value, e.g. 110
| |warm_up_epochs|Number of warm up epochs|int: any integer value, e.g. 10
| |warm_up_loss_function_path|Name of loss function during warm up epochs|string: classification loss eg. torch.nn.CrossEntropyLoss 
| |class_loss_function_path|Name of loss function for classification head|string: classification loss eg. torch.nn.CrossEntropyLoss 
| |adv_loss_function_path|Name of loss function for adversarial head(only for DCL)|string: classification loss eg. torch.nn.CrossEntropyLoss
| |jigsaw_loss_function_path|Name of loss function for reconstruction jigsaw head(only for DCL)|string: regression loss if prediction_type=regression eg. torch.nn.MSELoss, torch.nn.L1Loss, classification loss if prediction_type=class eg. torch.nn.BCELoss
| |rotation_loss_function_path|Name of loss function for rotation head(only for rotation)|string: classification loss eg. torch.nn.CrossEntropyLoss 
| |use_adv|Flag to indicate if adversarial loss of dcl must be used(only for DCL)|bool: True, False|
| |use_jigsaw|Flag to indicate if reconstruction loss of dcl must be used(only for DCL)|bool: True, False|
| |rotation_loss_weight|Factor that decides the contribution of rotation loss to total loss|float: Any value in the range [0,1]|
| |optimizer_path|Path to optimizer class|string: Path to optimizer, eg. torch.optim.SGD
| |optimizer_param: lr|Learning rate for parameter optimization|float: Any float value eg. 0.001|
| |optimizer_param: momentum|Momentum for optimizer(if SGD)|float: Any float value eg. 0.9
| |optimizer_param: weight_decay|Weight decay in optimizer|float: Any float value eg. 0.0001|
| |lr_scheduler: step_size|Step size of learning rate scheduler|int: any integer value. eg. 10, 30, 50|
| |lr_scheduler: gamma|Decay factor of learning rate scheduler|float: Any float value eg. 0.0001
