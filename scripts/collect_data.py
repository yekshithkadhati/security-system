import cv2
import os

def collect_images(label, num_images=50):
    print(f"Collecting images for label: {label}")
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open camera.")
        return
    cv2.namedWindow("Collecting Images")
    img_counter = 0
    os.makedirs(f'../data/train_set/{label}', exist_ok=True)
    print(f"Directory created: ../data/train_set/{label}")

    while img_counter < num_images:
        ret, frame = cam.read()
        if not ret:
            print("Failed to capture image")
            break
        cv2.imshow("Collecting Images", frame)
        k = cv2.waitKey(1)
        if k % 256 == 27:  # ESC pressed
            break
        elif k % 256 == 32:  # SPACE pressed
            img_name = f"../data/train_set/{label}/image_{img_counter}.png"
            cv2.imwrite(img_name, frame)
            print(f"{img_name} written!")
            img_counter += 1

    cam.release()
    cv2.destroyAllWindows()
    print("Collection completed")

if __name__ == "__main__":
    label = input("Enter the label (name) for the images: ")
    print(f"Label entered: {label}")
    collect_images(label)
