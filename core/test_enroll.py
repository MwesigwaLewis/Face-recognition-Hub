import cv2

from .enrollment import enroll_person


image = cv2.imread(
    "Kamagwa.jpg"
)


success, message = enroll_person(
    "KAMAGWA",
    image
)


print(message)