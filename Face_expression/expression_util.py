import cv2

class Visualizer:

    def __init__ (self):
        """
            
        """

    def draw_text_information(self, image, bounding_boxes, scores, predicteds, color = (40, 214, 63)):
        """
            This function draw lines of text describing information
        """
        font = cv2.FONT_HERSHEY_COMPLEX
        for i in range(len(bounding_boxes)):         
            txt = predicteds[i]
            cv2.putText(image, str(txt), (int(bounding_boxes[i][0]), int(bounding_boxes[i][1])), font, 0.25, color, 1, cv2.LINE_AA)