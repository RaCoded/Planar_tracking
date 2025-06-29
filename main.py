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
I0_pixels=tb.capture_reference("dossier_travail/Clown2.mp4","dossier_travail/")
I0_pixels = np.float32(I0_pixels)

# Définition des coordonnées du plan en 3D (mètres)
W_metre=np.array([[0,0],  #Coin en haut à gauche,
                      [0.21,0],  #Coin en haut à droite
                      [0,0.297],#Coin en bas à gauche
                      [0.210,0.297]])#Coin en bas à droite
# Coordonnées des points du repère 3D en mètres
axe=np.float32([[0,0,0,1],
                [0.10,0,0,1],
                [0,0.10,0,1],
                [0, 0, 0.10,1]])

# Chargement de l'image de référence pour affichage initial pré-image
img_ref = cv2.imread('dossier_travail/reference.jpg')  # Remplacez par le bon chemin de votre image


############################   Traitement repére monde <--> repére image  ###################



# Calcul de l'homographie entre le plan 3D et les points image en mètres
wH0, mask = cv2.findHomography(W_metre,I0_pixels , cv2.RANSAC, 2.0)
#Extraction de la matrice de translation de 0 vers w
wT0 = np.linalg.inv(mtx)@wH0
r1 = wT0[:,0]
r2 = wT0[:,1]
norm = np.linalg.norm(r1)
r1/=norm
r2/=norm
r3 = np.cross(r1,r2)
R = np.column_stack([r1, r2, r3])
t = wT0[:, 2] / norm
wT0 = mtx @ np.column_stack([R, t])

#On prépare l'affichage de l'axe
axeI0_homog = (wT0 @ axe.T).T # attention, il faut la transposée car une colonne = un point !
axeI0 = axeI0_homog[:, :-1] / axeI0_homog[:, -1][:, np.newaxis]
axeI0 = axeI0.astype(np.int32)
cv2.arrowedLine(img_ref, axeI0[0], axeI0[1] , (255,0,0),2)
cv2.arrowedLine(img_ref, axeI0[0], axeI0[2], (0,0,255),2)
cv2.arrowedLine(img_ref, axeI0[0], axeI0[3], (0,255,0),2)

#Test d'affichage
img_refCopy = cv2.resize(img_ref, (1000,1000))
cv2.imshow("test", img_refCopy)
cv2.waitKey(0)
cv2.destroyAllWindows()

############################   Détection et traitement des points SIFT dans la frame initiale  ###################
# Initialisation du détecteur SIFT
sift = cv2.SIFT_create()
# Détection des points d'intérêt et calcul des descripteurs SIFT sur l'image de référence
keypoints_reference, descriptors_reference = sift.detectAndCompute(img_ref, None)

# Filtrage des points SIFT pour ne garder que ceux dans le rectangle détecté
rect_points = np.array([[pt[0], pt[1]] for pt in I0_pixels], dtype=np.int32) #Rectangle dans lequel on cherche les descripteurs sift 
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
img_refCopy = cv2.resize(img_ref, (1000,1000))
cv2.imshow("test", img_refCopy)
cv2.waitKey(0)
cv2.destroyAllWindows()


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
fourcc = cv2.VideoWriter_fourcc(*'XVID')

output_video = cv2.VideoWriter('dossier_travail/output_video.avi', 
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
    print(fps)
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
    for point in img_points_sift:
        x, y = point
        # Dessine un cercle bleu à chaque point
        cv2.circle(img_frame, (int(x), int(y)), 5, (255, 0, 0), -1)
    #Homographie pour passer de la frame i-1 à la frame i courante
    previous_img_points_sift = np.float32([filtered_keypoints_pt_reference[m.queryIdx] for m in matches]).reshape(-1, 2)

    Hij, mask = cv2.findHomography(previous_img_points_sift,img_points_sift , cv2.RANSAC, 5.0)
    #Mise ç jour de l'homographie de la frame 0 à i
    H0i=np.dot(Hij,H0i)
    #On affiche le plan obtenu avec homographie
    img_points_transformed = cv2.perspectiveTransform(np.float32(I0_pixels).reshape(-1, 1, 2), H0i).reshape(-1, 2)
    points_rect_test = np.array([img_points_transformed[0], img_points_transformed[1], img_points_transformed[3], img_points_transformed[2]], dtype=np.int32)
    cv2.polylines(img_frame, [points_rect_test], isClosed=True, color=(0, 255, 0), thickness=2)
    
    #projection du repere 3d dans le 2d, on peut repasser ensuite aux metres si besoin
    Hwi=H0i@wH0  
    wTi = np.linalg.inv(mtx)@Hwi  
    r1 = wTi[:,0]
    r2 = wTi[:,1]
    norm = np.linalg.norm(r1)
    r1/=norm
    r2/=norm
    r3 = np.cross(r1,r2)
    R = np.column_stack([r1, r2, r3])
    t = wTi[:, 2] / norm
    wTi = mtx @ np.column_stack([R, t])
    #On prépare l'affichage de l'axe
    axeIi_homog = (wTi @ axe.T).T # attention, il faut la transposée car une colonne = un point !
    axeIi = axeIi_homog[:, :-1] / axeIi_homog[:, -1][:, np.newaxis]
    axeIi = axeIi.astype(np.int32)
    cv2.arrowedLine(img_frame, axeIi[0], axeIi[1] , (255,0,0),2)
    cv2.arrowedLine(img_frame, axeIi[0], axeIi[2], (0,0,255),2)
    cv2.arrowedLine(img_frame, axeIi[0], axeIi[3], (0,255,0),2)

  
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
    """  cv2.imshow("Tracking vidéo", cv2.resize(img_frame, (1000, 1000)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output_video.release()
cv2.destroyAllWindows()

end_time = time.time()
execution_time = end_time - start_time
print(f"Temps d'exécution : {execution_time} secondes")
