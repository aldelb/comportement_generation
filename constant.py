# --- type params
model_number = 0
unroll_steps =  False
layer = ""
hidden_size = 0
kernel_size = 0
padding_size = 0
dropout = 1

# --- Path params
datasets = ""
dir_path = ""
data_path = ""
saved_path = ""
output_path = ""
evaluation_path = ""
model_path = ""

# --- Training params
n_epochs =  0
batch_size = 0
d_lr =  0
g_lr =  0
log_interval =  0


# --- Data params
noise_size = 0
pose_size = 0 # nombre de colonne openface pose and gaze angle
au_size = 0 # nombre de colonne openface AUs
prosody_size = 0 #nombre de colonne opensmile selectionnées
derivative = False

opensmile_columns = []
selected_opensmile_columns = []
selected_os_index_columns = []
openface_columns = []
