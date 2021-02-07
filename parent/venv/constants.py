import json

with open('constants.txt', 'r') as f:
    config = json.load(f)
    SERVER = config["server"]
    REPOSITORY = config["repository"]
    PASSWORD = config["password"]
    USERNAME = config["username"]
    WORKING_DIRECTORY = config["working directory"]
    DOWNLOADED_IMAGE = config["downloaded image (name-prefix, its also the name of uploaded image)"]
    EAR_LABEL_IMAGE = config["image name for ear labels"]
    EINZELREIHE_LABEL_IMAGE = config["image name for einzelreihe labels"]
    PARZELLE_LABEL_IMAGE = config["image name for parzelle labels"]
    ONLINE_RESULTS_DIRECTORY = config["online annotation results directory"]
    FILE_FORMAT = config["file-format (for the label-image normally .bmp)"]
    INTERMEDIATE_PARZELLE = "intermediate_parzelle" + FILE_FORMAT
    INTERMEDIATE_EINZELREIHE = "intermediate_einzelreihe" + FILE_FORMAT
    FUSION_INTERMEDIATE_EAR = "ear_label_fusion" + FILE_FORMAT
    FUSION_INTERMEDIATE_PARZELLE = "parzelle_label_fusion" + FILE_FORMAT
    FUSION_INTERMEDIATE_EINZELREIHE = "einzelreihe_label_fusion" + FILE_FORMAT
    # HEALTHY_LABEL = config['healthy_label']  # healthy
    # DISEASED_LABEL = 90  # diseased
    EAR_LABEL = config["ear label"]
    SOIL_LABEL = config["soil label"]
    DOPPELREIHE_LABEL = config["doppelreihe label"]
    EINZELREIHE_LABEL = config["einzelreihe label"]
    PARZELLE_LABEL = config["parzelle label"]
    CANOPEO_RESULTS_PATH = config["results for canopeo"]
    K_MEANS_CENTERS = config["k-means centers path"]
    K_MEANS_RESULTS_PATH = config["k-means results path (annotation)"]
    REST_LABEL = config["rest label"]
    SOIL_LABEL_IMAGE = config["image name for soil labels"]
    REST_LABEL_IMAGE = config["image name for rest labels"]
    DISEASED_LABEL = config["diseased label"]
    HEALTHY_LABEL = config["healthy label"]
    DISEASED_LABEL_IMAGE = config["image name for diseased labels"]
    HEALTHY_LABEL_IMAGE = config["image name for healthy labels"]
    MODEL_NAME = config["name of model"]
    UNET_WORKING_DIR = config["unet working directory"]
if WORKING_DIRECTORY is None:  # just to test if everything is loaded
    print("There are missing constants in constants.txt!!!")
    exit(2)

ENCODING = "utf-8"
FUSION_INTERMEDIATE_H_D_L_I = "ear_label_fusion" + FILE_FORMAT
FUSION_INTERMEDIATE_D_S_L_I = "doppelreihe_soil_label_fusion" + FILE_FORMAT
### CANOPEO CONSTANTS
P1 = 0.95  # default: 0.95
P2 = 0.94  # default: 0.94
P3 = 20  # default: 20
KERNEL_SIZE = 3  # for refineing
### K-MEANS CONSTANTS
K_MEANS_CENTERS_CSV = "centers.csv"
K_MEANS_COLORS = "colors.csv"
K_MEANS_NR_CENTERS = 3
K_MEANS_NR_ITERATIONS = 1

UNET_RAW_PREDICTION = "raw_prediction.npy"
UNET_PREDICTION_IMG = "prediction.png"
UNET_RAW_MASK = "mask.png"
UNET_REFINED = "refined_prediction.png"
UNET_CURRENT_PREDICTION = "current_prediction.png"


"""
SERVER = "https://drive.boku.ac.at/"
REPOSITORY = "cac80769-e029-4f72-ab6c-73a70d9c231d"
PASSWORD = "pTaLqFqZ8hE3xCy"
USERNAME = "l.moser@student.tugraz.at"
ENCODING = "utf-8"
WORKING_DIRECTORY = ".tmp"
DOWNLOADED_IMAGE = "original"
EAR_LABEL_IMAGE = "ear_label.bmp"
DOPPELREIHE_SOIL_LABEL_IMAGE = "doppelreihe_soil_label.bmp"
INTERMEDIATE = "intermediate.bmp"
ONLINE_RESULTS_DIRECTORY = "annotation_results"
HEALTHY_LABEL = 50  # healthy
DISEASED_LABEL = 90  # diseased
EAR_LABEL = 50
SOIL_LABEL = 130
DOPPELREIHE_LABEL = 175
CANOPEO_RESULTS_PATH = '.results_canopeo'
K_MEANS_CENTERS = '.k_means_centers(dont_delete)'
K_MEANS_RESULTS_PATH = ".results_K_Means"
"""

#### KERAS

KERAS_WD = "keras"
RESIZED = "resized"
TRAINING_DATA = "train"
TEST_DATA = "test"