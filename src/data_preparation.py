import os 
import shutil
import cv2

Datasets = ['data/VisDrone2019-DET-train', 'data/VisDrone2019-DET-val']

Vis_Drone_to_YOLO_Class = {
    1: 0,  # piéton -> Humain
    2: 0,  # personne -> Humain
    3: 1,  # vélo -> Véhicule/Mobile
    4: 1,  # voiture -> Véhicule/Mobile
    5: 1,  # van -> Véhicule/Mobile
    6: 1,  # camion -> Véhicule/Mobile
    7: 1,  # tricycle -> Véhicule/Mobile
    8: 1,  # auvent de tricycle -> Véhicule/Mobile
    9: 1,  # bus -> Véhicule/Mobile
    10: 1, # moteur -> Véhicule/Mobile
}

def convert_to_yolo(split_name):
    """Convertit les annotations du dataset VisDrone au format YOLO."""
    print(f"Processing {split_name} dataset...")

    # Chemin de la source
    dataset_path = 'data/' + split_name
    annotation_path = os.path.join(dataset_path, 'annotations')
    images_path = os.path.join(dataset_path, 'images')

    #Chemin de la destination
    yolo_path = os.path.join('data', 'yolo_format', split_name.replace('VisDrone2019-DET-', ''))
    yolo_images_path = os.path.join(yolo_path, 'images')
    yolo_labels_path = os.path.join(yolo_path, 'labels')

    #Création des dossiers de destination
    os.makedirs(yolo_images_path, exist_ok=True)
    os.makedirs(yolo_labels_path, exist_ok=True)

    annotation_files = [f for f in os.listdir(annotation_path) if f.endswith('.txt')]

    for ann_file in annotation_files:
        input_ann_file = os.path.join(annotation_path, ann_file)
        yolo_ann_file = os.path.join(yolo_labels_path, ann_file)

        image_file = ann_file.replace('.txt', '.jpg')
        input_image_file = os.path.join(images_path, image_file)
        yolo_image_file = os.path.join(yolo_images_path, image_file)

        if not os.path.exists(input_image_file):
            print(f"Image file {input_image_file} not found, skipping...")
            continue
        
        shutil.copy(input_image_file, yolo_image_file)
        print(f"Copied image {input_image_file} to {yolo_image_file}")

        # Traitement du fichier des annotations
        with open(input_ann_file, 'r') as f_in, open(yolo_ann_file, 'w') as f_out:
            h,w = cv2.imread(input_image_file).shape[:2]
            for line in f_in:
                # VisDrone format: <category_id>, <bbox_left>, <bbox_top>, <bbox_width>, <bbox_height>, <score>, <truncation>, <occlusion>
                parts = line.strip().split(',')
                if len(parts) < 8:
                    print(f"Invalid annotation line in {input_ann_file}: {line.strip()}")
                    continue
                try:
                    category_id = int(parts[0])
                    bbox_left = float(parts[1])
                    bbox_top = float(parts[2])
                    bbox_width = float(parts[3])
                    bbox_height = float(parts[4])
                except ValueError:
                    print(f"Error parsing line in {input_ann_file}: {line.strip()}")
                    continue
                if category_id not in Vis_Drone_to_YOLO_Class:
                    continue  # Ignorer les catégories non mappées
                yolo_class_id = Vis_Drone_to_YOLO_Class[category_id]

                # Conversion au format YOLO en normalisé
                # YOLO format: <class_id> <x_center> <y_center> <width> <height>
                x_center = (bbox_left + bbox_width / 2) / w
                y_center = (bbox_top + bbox_height / 2) / h
                width = bbox_width / w
                height = bbox_height / h

                yolo_line = f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                f_out.write(yolo_line)

# Fonction principale
if __name__ == "__main__":
    yaml_config = """# YOLOv8 custom config for VisDrone-DET
path: ../data/yolo_format # Racine du dataset (relative à l'endroit où vous lancez l'entraînement)
train: train # Train images dir
val: val # Val images dir

# Classes
names:
  0: Humain
  1: Mobile_Obstacle 
"""
    with open("visdrone.yaml", "w") as f:
        f.write(yaml_config)
    for dataset in Datasets:
        convert_to_yolo(dataset.split('/')[-1])
    print("Data preparation completed.") 

