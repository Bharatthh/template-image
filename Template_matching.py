import cv2
import numpy as np
import os

class TemplateMatcher:
    def __init__(self, folder_path):
        self.matches_folders = {}
        self.template_images = []
        self.load_templates(folder_path)
        self.cap = cv2.VideoCapture(0)

    def load_templates(self, folder_path):
        for root, dirs, files in os.walk(folder_path):
            for directory in dirs:
                folder_path = os.path.join(root, directory)
                folder_name = os.path.basename(folder_path)
                template_images_folder = []

                for filename in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, filename)
                    if os.path.isfile(img_path):
                        template = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        template_images_folder.append(template)
                        self.matches_folders[id(template)] = folder_name

                self.template_images.extend(template_images_folder)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cap_frame = self.detect(frame_gray, frame)
            cv2.imshow('Camera', cap_frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def is_overlapping(self,R1,R2 ):
      if (R1[0]>=R2[2]) or (R1[2]<=R2[0]) or (R1[3]<=R2[1]) or (R1[1]>=R2[3]):
         return False
      return True

    def detect(self, frame_gray, frame, threshold=0.6):
        detected_rectangles = []
        for input_image in self.template_images:
            if input_image is not None:
                result = cv2.matchTemplate(frame_gray, input_image, cv2.TM_CCOEFF_NORMED)

                locations = np.where(result >= threshold)
                for loc in zip(*locations[::-1]):
                    folder_name = self.matches_folders[id(input_image)]
                    h, w = input_image.shape[:2]
                    rect = (loc[0], loc[1], w, h)

                    is_overlapping = any(self.is_overlapping(rect, existing_rect) for existing_rect in detected_rectangles)
                    if not is_overlapping:
                        cv2.rectangle(frame, loc, (loc[0] + w, loc[1] + h), (0, 255, 0), 3)
                        cv2.putText(frame, folder_name, (loc[0], loc[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        detected_rectangles.append(rect)
        return frame

if __name__ == "__main__":
    folder_path = r"C:\Users\Bharath\Desktop\template_matching\images"
    template_matcher = TemplateMatcher(folder_path)
    template_matcher.run()
