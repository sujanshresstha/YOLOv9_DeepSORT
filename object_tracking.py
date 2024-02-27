import cv2
import torch
import numpy as np
from absl import app, flags
from absl.flags import FLAGS
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape

# Define command line flags
flags.DEFINE_string('video', './data/test.mp4', 'Path to input video or webcam index (0)')
flags.DEFINE_string('output', './output/output.mp4', 'path to output video')
flags.DEFINE_float('conf', 0.50, 'confidence threshold')
flags.DEFINE_integer('blur_id', None, 'class ID to apply Gaussian Blur')
flags.DEFINE_integer('class_id', None, 'class ID to track')

def main(_argv):
  # Initialize the video capture
  video_input = FLAGS.video
  # Check if the video input is an integer (webcam index)
  if FLAGS.video.isdigit():
      video_input = int(video_input)
      cap = cv2.VideoCapture(video_input)
  else:
      cap = cv2.VideoCapture(video_input)
  if not cap.isOpened():
      print('Error: Unable to open video source.')
      return
  frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = int(cap.get(cv2.CAP_PROP_FPS))
  # video writer objects
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  writer = cv2.VideoWriter(FLAGS.output, fourcc, fps, (frame_width, frame_height))

  # Initialize the DeepSort tracker
  tracker = DeepSort(max_age=50)
  # select device (CPU or GPU)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Load YOLO model
  model = DetectMultiBackend(weights='./weights/yolov9-e.pt',device=device, fuse=True)
  model = AutoShape(model)

  # Load the COCO class labels
  classes_path = "../configs/coco.names"
  with open(classes_path, "r") as f:
      class_names = f.read().strip().split("\n")

  # Create a list of random colors to represent each class
  np.random.seed(42)
  colors = np.random.randint(0, 255, size=(len(class_names), 3)) 

  while True:
      ret, frame = cap.read()
      if not ret:
          break
      # Run model on each frame
      results = model(frame)
      detect = []
      for det in results.pred[0]:
          label, confidence, bbox = det[5], det[4], det[:4]
          x1, y1, x2, y2 = map(int, bbox)
          class_id = int(label)

          # Filter out weak detections by confidence threshold and class_id
          if FLAGS.class_id is None:
              if confidence < FLAGS.conf:
                  continue
          else:
              if class_id != FLAGS.class_id or confidence < FLAGS.conf:
                  continue

          detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

      tracks = tracker.update_tracks(detect, frame=frame)

      for track in tracks:
          if not track.is_confirmed():
              continue
          track_id = track.track_id
          ltrb = track.to_ltrb()
          class_id = track.get_det_class()
          x1, y1, x2, y2 = map(int, ltrb)
          color = colors[class_id]
          B, G, R = map(int, color)
          text = f"{track_id} - {class_names[class_id]}"

          cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
          cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 12, y1), (B, G, R), -1)
          cv2.putText(frame, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

          # Apply Gaussian Blur
          if FLAGS.blur_id is not None and class_id == FLAGS.blur_id:
              if 0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]:
                  frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (99, 99), 3)

      cv2.imshow('YOLOv9 Object tracking', frame)
      writer.write(frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

  # Release video capture and writer
  cap.release()
  writer.release()

if __name__ == '__main__':
  try:
      app.run(main)
  except SystemExit:
      pass
