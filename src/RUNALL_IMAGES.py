import straighten_faces
import detect_landmarks
import morph_faces
import get_components

straighten_faces.main()
detect_landmarks.get_landmarks()
detect_landmarks.find_average_face()
morph_faces.warp_faces()
morph_faces.shapeless_intensity()
get_components.main()
