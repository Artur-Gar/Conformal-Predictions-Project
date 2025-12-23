import os

# absolute base directory of upstream_xray_classifier/ (where this config.py lives)
_base_dir = os.path.dirname(os.path.abspath(__file__))

# make all dirs absolute (independent of working directory)
pkl_dir_path = os.path.join(_base_dir, "pickles")
models_dir   = os.path.join(_base_dir, "models")

# keep filenames as-is (these are joined with pkl_dir_path elsewhere)
train_val_df_pkl_path    = "train_val_df.pickle"
test_df_pkl_path         = "test_df.pickle"
disease_classes_pkl_path = "disease_classes.pickle"

from torchvision import transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# transforms.RandomHorizontalFlip() not used because some disease might be more likely to the present in a specific lung (lelf/rigth)
transform = transforms.Compose([transforms.ToPILImage(), 
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    normalize])
