import cv2
import numpy as np
import time
import Traitement_basique as tb

start_time = time.time()

############################   Données connues   ###############################

# Chargement de la matrice de calibration de la caméra obtenue par OpenCV
calibration_data = np.load('dossier_calibration/camera_calibration.npz')
mtx = calibration_data['mtx'] 
dist = calibration_data['dist']  

# Définition manuelle de la matrice de calibration
mtx=np.array([[1.60478121e+03 ,0.00000000e+00 ,5.67241007e+02],
 [0.00000000e+00 ,1.60478121e+03 ,9.87425440e+02],
 [0.00000000e+00 ,0.00000000e+00 ,1.00000000e+00],])


# Capture des points dans le repére image sur l'image de référence
img_points_pixel=tb.capture_reference("dossier_travail/Clown2.mp4","dossier_travail/")
img_points_pixel = np.float32(img_points_pixel)
# Conversion des points image en coordonnées homogènes
img_points_pixel_extend=np.hstack([img_points_pixel, np.ones((img_points_pixel.shape[0], 1))])
# Conversion des coordonnées en mètres
img_points_pixel_extend_metre=np.dot(np.linalg.inv(mtx), img_points_pixel_extend.T).T

# Définition des coordonnées du plan en 3D (mètres)
world_points_plan=np.array([[0,0],  #Coin en haut à gauche,
                      [0.21,0],  #Coin en haut à droite
                      [0,0.297],#Coin en bas à gauche
                      [0.210,0.297]])#Coin en bas à droite
# Coordonnées des points du repère 3D en mètres
ref_points=np.float32([[0.1,0,0],
                     [0,0.1,0],
                     [0,0,0],
                     [0, 0, 0.1]])

# Chargement de l'image de référence pour affichage initial pré-image
img_ref = cv2.imread('dossier_travail/reference.jpg')  # Remplacez par le bon chemin de votre image


############################   Traitement repére monde <--> repére image  ###################



# Calcul de l'homographie entre le plan 3D et les points image en mètres
H_metre_frame0, mask = cv2.findHomography(world_points_plan,img_points_pixel_extend_metre , cv2.RANSAC, 2.0)
# Transformation en pixels en multipliant par la matrice de calibration
H_frame0=np.dot(mtx, H_metre_frame0)

# Transformation des points du plan avec l'homographie
img_points_transformed = cv2.perspectiveTransform(np.float32(world_points_plan).reshape(-1, 1, 2), H_frame0).reshape(-1, 2)
img_points_transformed[:, [0, 1]] = img_points_transformed[:, [1, 0]]  # Inversion pour OpenCV afin d'avoire (colonne,ligne)

# Dessin du rectangle correspondant au plan détecté
points_rect_test = np.array([img_points_transformed[0], img_points_transformed[1], img_points_transformed[3], img_points_transformed[2]], dtype=np.int32)
cv2.polylines(img_ref, [points_rect_test], isClosed=True, color=(0, 255, 0), thickness=2)

# Transformation du repére 3d en 2d
c1=H_metre_frame0[:,0]
c2=H_metre_frame0[:,1]
c3=np.cross(c1, c2)
R =np.column_stack((c1, c2, c3))
t=H_metre_frame0[:,2]
pose_points2d = [np.dot(mtx, (R @ x + t).T).T for x in ref_points]
#Ici, le w n'est pas forcément égal à 1 car on ne calque pas un plan mais un objet qui n'est pas adapté à des homographies. On peut seulement estimer la pose du repére 3d par rapport à l'image. On normalise donc les points en divisant par Z sauf pour le dernier point (qui est à l'infini)
pose_points2d_normalized = []
for i, p in enumerate(pose_points2d):
    if i == len(pose_points2d) - 1:  # Dernier point
        pose_points2d_normalized.append(p[:2])  # Ne pas diviser par Z (dernier point en dehors du plan)
    else:
        pose_points2d_normalized.append(p[:2]/p[2] )  # Diviser par Z pour les autres points
pose_points2d=pose_points2d_normalized

#on prépare l'affichage puis on affiche
pose_points2d = [tuple(np.int32(p[:2])) for p in pose_points2d]
cv2.arrowedLine(img_ref, (pose_points2d[2][1],pose_points2d[2][0]), (pose_points2d[0][1],pose_points2d[0][0]), (255,0,0),2)
cv2.arrowedLine(img_ref, (pose_points2d[2][1],pose_points2d[2][0]), (pose_points2d[1][1],pose_points2d[1][0]), (0,0,255),2)
cv2.arrowedLine(img_ref, (pose_points2d[2][1],pose_points2d[2][0]), (pose_points2d[3][1],pose_points2d[3][0]), (0,255,0),2)



############################   Détection et traitement des points SIFT dans la frame initiale  ###################
# Initialisation du détecteur SIFT
sift = cv2.SIFT_create()
# Détection des points d'intérêt et calcul des descripteurs SIFT sur l'image de référence
keypoints_reference, descriptors_reference = sift.detectAndCompute(img_ref, None)

# Filtrage des points SIFT pour ne garder que ceux dans le rectangle détecté
rect_points = np.array([[pt[1], pt[0]] for pt in img_points_pixel], dtype=np.int32) #Rectangle dans lequel on cherche les descripteurs sift 
filtered_keypoints_pt_reference = [] 
filtered_descriptors_reference = [] 
for keypoint,descriptor in zip(keypoints_reference,descriptors_reference):
    x, y = keypoint.pt  
    if cv2.pointPolygonTest(rect_points, (x, y), False) >= 0:
        filtered_keypoints_pt_reference.append((keypoint.pt))
        filtered_descriptors_reference.append(descriptor)
filtered_keypoints_pt_reference=np.array(filtered_keypoints_pt_reference)
filtered_descriptors_reference=np.array(filtered_descriptors_reference)

#dessiner les sift souus forme de points verts
for point in filtered_keypoints_pt_reference:
        x, y = point
        # Dessine un cercle bleu à chaque point
        cv2.circle(img_ref, (int(x), int(y)), 3, (0, 255, 0), -1)
#img_with_keypoints = cv2.drawKeypoints(img_ref, filtered_keypoints_reference, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



############################   Traitement de la vidéo   ###################
##################################################################################



#stockage de l'homographie permettant de passer de la frame 0 à la frame i courante
H0i=np.identity(3)
# On lance la vidéo
cap = cv2.VideoCapture('dossier_travail/Clown2.mp4')
# Un matcher de points sift, peut être que flann serait mieux..
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')

output_video = cv2.VideoWriter('dossier_travail/output_video.mp4', 
                               fourcc, 
                               fps, 
                               (width, height))

#Parcourt des frames
cap.read() #on passe la premiére frame qui est déjà lue
prev_frame_time = 0 #de quoi calculer le temps d'execution
new_frame_time = 0
while cap.isOpened():
    ret, img_frame = cap.read()
    
    #Vérification qu'one est dans la vidéo
    if not ret:
        break

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)  # FPS = 1 / temps écoulé
    prev_frame_time = new_frame_time  # Met à jour le temps précédent
    # Afficher les FPS sur l'image
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(img_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    #Detection des points sift dans la frame ouverte
    keypoints_frame, descriptors_frame = sift.detectAndCompute(img_frame, None)
    #Matching de ces sift avec ceux de la frame 0 et trie pour garder les plus proches
    matches = bf.match(filtered_descriptors_reference, descriptors_frame)
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:]  # au cas où on souhaite pas garder toutes les correspondances

    #Obtention des correspondances de point
    img_points_sift = np.float32([keypoints_frame[m.trainIdx].pt for m in matches]).reshape(-1, 2) #points sift dans l'image courante
    img_points_sift[:, [0, 1]] = img_points_sift[:, [1, 0]] #sous la forme (ligne,colonne) pour traitement
    for point in img_points_sift:
        x, y = point
        # Dessine un cercle bleu à chaque point
        cv2.circle(img_frame, (int(y), int(x)), 5, (255, 0, 0), -1)
    #Homographie pour passer de la frame i-1 à la frame i courante
    previous_img_points_sift = np.float32([filtered_keypoints_pt_reference[m.queryIdx] for m in matches]).reshape(-1, 2)
    previous_img_points_sift[:, [0, 1]] = previous_img_points_sift[:, [1, 0]] #sous la forme (ligne,colonne) pour traitement

    Hij, mask = cv2.findHomography(previous_img_points_sift,img_points_sift , cv2.RANSAC, 5.0)
    #Mise ç jour de l'homographie de la frame 0 à i
    H0i=np.dot(Hij,H0i)
    #On affiche le plan obtenu avec homographie
    img_points_transformed = cv2.perspectiveTransform(np.float32(img_points_pixel).reshape(-1, 1, 2), H0i).reshape(-1, 2)
    img_points_transformed[:, [0, 1]] = img_points_transformed[:, [1, 0]] #sous la forme (colonne,ligne) pour opencv
    points_rect_test = np.array([img_points_transformed[0], img_points_transformed[1], img_points_transformed[3], img_points_transformed[2]], dtype=np.int32)
    cv2.polylines(img_frame, [points_rect_test], isClosed=True, color=(0, 255, 0), thickness=2)
    
    #projection du repere 3d dans le 2d, on peut repasser ensuite aux metres si besoin
    Hwi=H0i@H_frame0    
    c1=Hwi[:,0]
    c2=Hwi[:,1]
    c3=np.cross(c1, c2)
    R =np.column_stack((c1, c2, c3))
    t=Hwi[:,2]
    pose_points2d=[(R@x + t).T for x in (ref_points)]
    pose_points2d_normalized = []
    for i, p in enumerate(pose_points2d):
        if i == len(pose_points2d) - 1:  # Dernier point
            pose_points2d_normalized.append(p[:2])  # Ne pas diviser par Z (dernier point)
        else:
            pose_points2d_normalized.append(p[:2] / p[2])  # Diviser par Z pour les autres points
    pose_points2d=pose_points2d_normalized
    pose_points2d = [tuple(np.int32(p[:2])) for p in pose_points2d]
    #On dessine le repere
    cv2.arrowedLine (img_frame, (pose_points2d[2][1],pose_points2d[2][0]), (pose_points2d[0][1],pose_points2d[0][0]), (255,0,0),2)
    cv2.arrowedLine (img_frame, (pose_points2d[2][1],pose_points2d[2][0]), (pose_points2d[1][1],pose_points2d[1][0]), (0,0,255),2)
    cv2.arrowedLine (img_frame, (pose_points2d[2][1],pose_points2d[2][0]), (pose_points2d[3][1],pose_points2d[3][0]), (0,255,0),2)

    #écriture de la frame dans la vidéo d'output
    output_video.write(img_frame) 

    #On met à jour les points sift et les descripteurs sift précédents sous la condition qu'ils sont dans le cadre prédit (bon prédicat?)
    filtered_descriptors_reference = []
    filtered_keypoints_pt_reference = []
    for keypoint,descriptor in zip(keypoints_frame,descriptors_frame):
        x, y = keypoint.pt  
        if cv2.pointPolygonTest(points_rect_test, (x, y), False) >= 0:
            filtered_keypoints_pt_reference.append((keypoint.pt))
            filtered_descriptors_reference.append(descriptor)
    filtered_descriptors_reference = np.array(filtered_descriptors_reference, dtype=np.float32)
    filtered_keypoints_pt_reference = np.array(filtered_keypoints_pt_reference, dtype=np.float32)

    # Sortie si l'utilisateur appuie sur 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output_video.release()
cv2.destroyAllWindows()

end_time = time.time()
execution_time = end_time - start_time
print(f"Temps d'exécution : {execution_time} secondes")
