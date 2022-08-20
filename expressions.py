import face_recognition
from PIL import Image, ImageDraw

# This is an example of running face recognition on a single image
# and drawing a box around each person that was identified.

# Load a sample picture and learn how to recognize it.
neutral_image = face_recognition.load_image_file("neutral.jpg")
neutral_face_encoding = face_recognition.face_encodings(neutral_image)[0]

# Load a second sample picture and learn how to recognize it.
sad_image = face_recognition.load_image_file("sad.jpg")
sad_face_encoding = face_recognition.face_encodings(sad_image)[0]

fear_image = face_recognition.load_image_file("fear.jpg")
fear_face_encoding = face_recognition.face_encodings(fear_image)[0]

disgust_image = face_recognition.load_image_file("disgust.jpg")
disgust_face_encoding = face_recognition.face_encodings(disgust_image)[0]
happy_image = face_recognition.load_image_file("happy.jpg")
happy_face_encoding = face_recognition.face_encodings(happy_image)[0]
angry_image = face_recognition.load_image_file("angry.jpg")
angry_face_encoding = face_recognition.face_encodings(angry_image)[0]
surprize_image = face_recognition.load_image_file("surprize.jpg")
surprize_face_encoding = face_recognition.face_encodings(surprize_image)[0]
# Create arrays of known face encodings and their names
known_face_encodings = [
    neutral_face_encoding,
    sad_face_encoding,
    fear_face_encoding,
    disgust_face_encoding,
    happy_face_encoding,
    angry_face_encoding,
    surprize_face_encoding
]
known_face_names = [
    "neutral",
    "sad",
    "happy",
    "surprize","disgust","fear","angry"

]

# Load an image with an unknown face
unknown_image = face_recognition.load_image_file("exp.jpg")

# Find all the faces and face encodings in the unknown image
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

# Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
# See http://pillow.readthedocs.io/ for more about PIL/Pillow
pil_image = Image.fromarray(unknown_image)
# Create a Pillow ImageDraw Draw instance to draw with
draw = ImageDraw.Draw(pil_image)

# Loop through each face found in the unknown image
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "Unknown"

    # If a match was found in known_face_encodings, just use the first one.
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]

    # Draw a box around the face using the Pillow module
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

    # Draw a label with a name below the face
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))


# Remove the drawing library from memory as per the Pillow docs
del draw

# Display the resulting image
pil_image.show()

# You can also save a copy of the new image to disk if you want by uncommenting this line
# pil_image.save("image_with_boxes.jpg")
