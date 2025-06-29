import numpy as np
import cv2
import glob


import cv2
import os

def convert_video_into_images(video_path, output_folder, max_frame=20,frame_interval=22):
    """
    Extrait une frame sur dix d'une vidéo et les enregistre dans un dossier.

    :param video_path: Chemin vers le fichier vidéo (.mp4)
    :param output_folder: Chemin vers le dossier où sauvegarder les frames
    :param frame_interval: Intervalle de frames à sauvegarder (par défaut 10)
    """
    # Ouvre la vidéo
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0  # Compteur de frames
    saved_count = 0  # Compteur de frames sauvegardées

    while True:
        ret, frame = cap.read()
        if not ret or saved_count>max_frame:  # Fin de la vidéo
            break

        # Si la frame est la n-ième frame à sauvegarder
        if frame_count % frame_interval == 0:
            # Nom de fichier basé sur le compteur de frames
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
            print(f"Frame {frame_count} sauvegardée sous {frame_filename}")

        frame_count += 1

    cap.release()  # Libère la vidéo
    print(f"Extraction terminée : {saved_count} frames sauvegardées dans {output_folder}")

#convert_video_into_images("dossier_calibration/video_calibration.mp4","dossier_calibration/")



# Dimensions du damier = (colonnes, lignes)
CHECKERBOARD = (7, 7)  
# Taille d'un carré du damier en cm
square_size = 1.3  

def calibration(CHECKERBOARD,square_size):
    # Préparation des points d sous la forme Nx2
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * square_size

    # Vecteurs pour stocker les points d'objet et les points d'image
    objpoints = []  # Points 3D dans le monde réel
    imgpoints = []  # Points 2D dans l'image

    # Charger les images
    images = glob.glob('dossier_calibration/*.jpg')  # images pour la calibration
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Trouver les coins du damier
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        # Si des coins sont trouvés, les ajouter aux listes
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            # Afficher les coins trouvés sur l'image
            cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
            #Pour l'affichage car relou
            scale_percent = 50 # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            resized = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
            cv2.imshow('Image', resized)
            cv2.waitKey()  # Attendre 

    cv2.destroyAllWindows()

    # Calibration de la caméra
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Afficher les paramètres de calibration
    print("Matrice de calibration :\n", mtx)
    print("Coefficients de distorsion :\n", dist)
    # Enregistrer les paramètres dans un fichier
    print("rotation ", rvecs)
    print("translation", tvecs)
    np.savez('dossier_calibration/camera_calibration.npz', mtx=mtx, dist=dist)

    return mtx,dist


#calibration(CHECKERBOARD,square_size)
