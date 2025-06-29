import cv2
import numpy as np

video_path="dossier_travail/"


def capture_image_ref(image, output_dir, scale_percent=40):
    """
    Affiche une image redimensionnée à scale_percent%,
    permet de cliquer sur cette image redimensionnée,
    et retourne les coordonnées des points dans l'image originale (non redimensionnée).
    """
    # Calcul des dimensions de l'image redimensionnée
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    resized = cv2.resize(image.copy(), (width, height), interpolation=cv2.INTER_AREA)

    image_points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # On capture les coords originales
            orig_x = int(x * 100 / scale_percent)
            orig_y = int(y * 100 / scale_percent)

            image_points.append((orig_x, orig_y))  

            # Dessiner le cercle sur l'image redimensionnée
            cv2.circle(resized, (x, y), 5, (0, 255, 0), -1)

            # Afficher les coordonnées originales sur l'image redimensionnée
            text = f"({orig_x}, {orig_y})"
            cv2.putText(resized, text, (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Image", resized)

    cv2.imshow("Image", resized)
    cv2.setMouseCallback("Image", click_event)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    return image_points


def capture_reference(video_path, output_dir):
    """
    Récupère la première frame d’une vidéo, affiche l'image en taille originale,
    permet de cliquer pour récupérer des points, et retourne les coordonnées originales (sans rescale).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("La vidéo n'est pas ouverte")
        return []

    ret, frame = cap.read()
    if not ret:
        print("Erreur lors de la lecture de la vidéo")
        cap.release()
        return []

    # Enregistrer l’image de référence en taille originale
    cv2.imwrite(output_dir + "reference.jpg", frame)

    # Appeler capture_image_ref avec la frame originale
    image_points = capture_image_ref(frame, output_dir)

    cap.release()
    cv2.destroyAllWindows()

    print("Points monde dans l'image de référence (coordonnées originales):", image_points)
    return image_points