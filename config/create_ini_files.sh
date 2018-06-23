cat <<EOT >> "data_path.ini"
[sample]
train_image_paths = ../sample_data/train_image_paths.csv
train_labeled_studies = ../sample_data/train_labeled_studies.csv
valid_image_paths = ../sample_data/valid_image_paths.csv
valid_labeled_studies = ../sample_data/valid_labeled_studies.csv
train_imgs = ../sample_data/train/
valid_imgs = ../sample_data/valid/
[csv]
train_image_paths = <enter/path>
train_labeled_studies = <enter/path>
valid_image_paths = <enter/path>
valid_labeled_studies = <enter/path>
[img]
train_imgs = <enter/path>
valid_imgs = <enter/path>
EOT
