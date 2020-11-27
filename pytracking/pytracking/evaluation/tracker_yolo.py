import importlib
import os
import numpy as np
from collections import OrderedDict
from pytracking.evaluation.environment import env_settings
import time
import cv2 as cv
from pytracking.utils.visdom import Visdom
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pytracking.utils.plotting import draw_figure, overlay_mask
from pytracking.utils.convert_vot_anno_to_rect import convert_vot_anno_to_rect
from ltr.data.bounding_box_utils import masks_to_bboxes
from pytracking.evaluation.multi_object_wrapper import MultiObjectWrapper
from pathlib import Path
import torch

_tracker_disp_colors = {1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 0, 0),
                        4: (255, 255, 255), 5: (0, 0, 0), 6: (0, 255, 128),
                        7: (123, 123, 123), 8: (255, 128, 0), 9: (128, 0, 255)}


def trackerlist(name: str, parameter_name: str, run_ids = None, display_name: str = None):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [TrackerYolo(name, parameter_name, run_id, display_name) for run_id in run_ids]


class TrackerYolo:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, run_id: int = None, display_name: str = None):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.run_id = run_id
        self.display_name = display_name

        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
            self.segmentation_dir = '{}/{}/{}'.format(env.segmentation_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
            self.segmentation_dir = '{}/{}/{}_{:03d}'.format(env.segmentation_path, self.name, self.parameter_name, self.run_id)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tracker', self.name))
        if os.path.isdir(tracker_module_abspath):
            tracker_module = importlib.import_module('pytracking.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

        self.visdom = None


    def _init_visdom(self, visdom_info, debug):
        visdom_info = {} if visdom_info is None else visdom_info
        self.pause_mode = False
        self.step = False
        if debug > 0 and visdom_info.get('use_visdom', True):
            try:
                self.visdom = Visdom(debug, {'handler': self._visdom_ui_handler, 'win_id': 'Tracking'},
                                     visdom_info=visdom_info)

                # Show help
                help_text = 'You can pause/unpause the tracker by pressing ''space'' with the ''Tracking'' window ' \
                            'selected. During paused mode, you can track for one frame by pressing the right arrow key.' \
                            'To enable/disable plotting of a data block, tick/untick the corresponding entry in ' \
                            'block list.'
                self.visdom.register(help_text, 'text', 1, 'Help')
            except:
                time.sleep(0.5)
                print('!!! WARNING: Visdom could not start, so using matplotlib visualization instead !!!\n'
                      '!!! Start Visdom in a separate terminal window by typing \'visdom\' !!!')

    def _visdom_ui_handler(self, data):
        if data['event_type'] == 'KeyPress':
            if data['key'] == ' ':
                self.pause_mode = not self.pause_mode

            elif data['key'] == 'ArrowRight' and self.pause_mode:
                self.step = True


    def create_tracker(self, params):
        tracker = self.tracker_class(params)
        tracker.visdom = self.visdom
        return tracker

    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """

        # START YOLO
        # load the COCO class labels our YOLO model was trained on
        # and the classes that wont be used (coco.names contains the names)
        # Init a detection flag
        det_flag = 0
        labelsPath = "/home/ebo/Parrot_CV_Project/local_yolo/yolo-coco/coco.names"
        LABELS = open(labelsPath).read().strip().split("\n")
        outdoor_classes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                           17, 18, 19, 20, 21, 22, 23,
                           29, 30, 31, 32, 33, 34, 35, 36, 37, 38]

        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

        # derive the paths to the YOLO weights and model configuration
        weightsPath = "/home/ebo/Parrot_CV_Project/local_yolo/yolo-coco/yolov3.weights"
        configPath = "/home/ebo/Parrot_CV_Project/local_yolo/yolo-coco/yolov3.cfg"

        # load our YOLO object detector trained on COCO dataset (80 classes)
        print("[INFO] loading YOLO from disk...")
        net = cv.dnn.readNetFromDarknet(configPath, weightsPath)

        # set CUDA backend
        # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # determine only the output layers from yolo
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        (W, H) = (None, None)
        # END YOLO INIT

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)
            if hasattr(tracker, 'initialize_features'):
                tracker.initialize_features()

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        output_boxes = []

        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + tracker.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        success, frame = cap.read()
        cv.imshow(display_name, frame)

        def _build_init_info(box):
            return {'init_bbox': OrderedDict({1: box}), 'init_object_ids': [1, ], 'object_ids': [1, ],
                    'sequence_object_ids': [1, ]}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
        else:
            while True:
                # cv.waitKey()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                           1.5, (0, 0, 0), 1)
                # HERE WE MUST HAVE YOLO operating
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)
                break

        while True:
            ret, frame = cap.read()

            if frame is None:
                break

            frame_disp = frame.copy()

            # YOLO IN
            # if the frame dimensions are empty, grab them
            frame_yolo = frame.copy()
            if W is None or H is None:
                (H, W) = frame_yolo.shape[:2]

            # construct a blob from the input frame and then perform a forward
            # pass of the YOLO object detector, giving us our bounding boxes
            # and associated probabilities
            blob = cv.dnn.blobFromImage(frame_yolo, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            layerOutputs = net.forward(ln)

            # initialize our lists of detected bounding boxes, confidences,
            # and class IDs, respectively
            boxes = []
            confidences = []
            classIDs = []

            # loop over each of the layer outputs
            for output in layerOutputs:
                # loop over each of the detections
                for detection in output:
                    # extract the class ID and confidence (i.e., probability)
                    # of the current object detection
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    # filter weak prediction and unrelated classes
                    if classID not in outdoor_classes and confidence > 0.5:
                        # scale the bounding box coordinates back relative to
                        # the size of the image, keeping in mind that YOLO
                        # actually returns the center (x, y)-coordinates of
                        # the bounding box followed by the boxes' width and
                        # height
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        # use the center (x, y)-coordinates to derive the top
                        # and and left corner of the bounding box
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        # update our list of bounding box coordinates,
                        # confidences, and class IDs
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            # apply non-maxima suppression to suppress weak, overlapping
            # bounding boxes
            idxs = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

            # ensure at least one detection exists
            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    # draw a bounding box rectangle and label on the frame
                    color = [int(c) for c in COLORS[classIDs[i]]]
                    cv.rectangle(frame_yolo, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                    cv.putText(frame_yolo, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    if classIDs[i] == 0:
                        det_flag = 1
                        tl_yolo = (x, y)  # top left coordinates
                        br_yolo = ((x + w), (y + h))  # bottom right coordinates
                        coordinates_text = "{} {}".format(tl_yolo, br_yolo)
                        cv.rectangle(frame_yolo, tl_yolo, br_yolo, (255, 255, 255), 2)
            """
            # if there's a path, write the info
            if args["output"] != 0:
                # check if the video writer is None
                if writer is None:
                    # initialize our video writer
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                             (frame.shape[1], frame.shape[0]), True)

                # write the output frame to disk
                writer.write(frame)
            """
            # YOLO OUT

            # Draw box
            out = tracker.track(frame)
            state = [int(s) for s in out['target_bbox'][1]]
            output_boxes.append(state)

            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)

            font_color = (0, 0, 0)
            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)

            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ret, frame = cap.read()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                           (0, 0, 0), 1)

                cv.imshow(display_name, frame_disp)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')

    def run_webcam(self, debug=None, visdom_info=None):
        """Run the tracker with the webcam.
        args:
            debug: Debug level.
        """

        def yolo_search(W, H, frame_yolo):
            fl = 0
            # if the frame dimensions are empty, grab them
            if W is None or H is None:
                (H, W) = frame_yolo.shape[:2]

            # construct a blob from the input frame and then perform a forward
            # pass of the YOLO object detector, giving us our bounding boxes
            # and associated probabilities
            blob = cv.dnn.blobFromImage(frame_yolo, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            layerOutputs = net.forward(ln)

            # initialize our lists of detected bounding boxes, confidences,
            # and class IDs, respectively
            boxes = []
            confidences = []
            classIDs = []

            # loop over each of the layer outputs
            for output in layerOutputs:
                # loop over each of the detections
                for detection in output:
                    # extract the class ID and confidence (i.e., probability)
                    # of the current object detection
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    # filter weak prediction and unrelated classes
                    if classID not in outdoor_classes and confidence > 0.5:
                        # scale the bounding box coordinates back relative to
                        # the size of the image, keeping in mind that YOLO
                        # actually returns the center (x, y)-coordinates of
                        # the bounding box followed by the boxes' width and
                        # height
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        # use the center (x, y)-coordinates to derive the top
                        # and and left corner of the bounding box
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        # update our list of bounding box coordinates,
                        # confidences, and class IDs
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            # apply non-maxima suppression to suppress weak, overlapping
            # bounding boxes
            idxs = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

            # ensure at least one detection exists
            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    # draw a bounding box rectangle and label on the frame
                    color = [int(c) for c in COLORS[classIDs[i]]]
                    cv.rectangle(frame_yolo, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                    cv.putText(frame_yolo, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    if classIDs[i] == 0:
                        detection_flag = 1
                        tl_coor = (x, y)  # top left coordinates
                        br_coor = ((x + w), (y + h))  # bottom right coordinates
                        coordinates_text = "{} {}".format(tl_coor, br_coor)
                        cv.rectangle(frame_yolo, tl_coor, br_coor, (255, 255, 255), 2)
                        fl = 1
                        return tl_coor, br_coor, detection_flag, frame_yolo
            if fl == 0:
                return (0, 0), (0, 0), 0, frame_yolo

        # load the COCO class labels our YOLO model was trained on
        # and the classes that wont be used (coco.names contains the names)
        # Init a detection flag
        det_flag = 0
        labelsPath = "/home/ebo/Parrot_CV_Project/local_yolo/yolo-coco/coco.names"
        LABELS = open(labelsPath).read().strip().split("\n")
        outdoor_classes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                           17, 18, 19, 20, 21, 22, 23,
                           29, 30, 31, 32, 33, 34, 35, 36, 37, 38]

        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

        # derive the paths to the YOLO weights and model configuration
        weightsPath = "/home/ebo/Parrot_CV_Project/local_yolo/yolo-coco/yolov3.weights"
        configPath = "/home/ebo/Parrot_CV_Project/local_yolo/yolo-coco/yolov3.cfg"

        # load our YOLO object detector trained on COCO dataset (80 classes)
        print("[INFO] loading YOLO from disk...")
        net = cv.dnn.readNetFromDarknet(configPath, weightsPath)

        # determine only the output layers from yolo
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        (W, H) = (None, None)
        temp_flag = 0

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name

        self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)
        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        class UIControl:
            def __init__(self):
                self.mode = 'init'  # init, select, track
                self.new_init = False

            def get_bb(self):
                # yolo bb
                if det_flag == 1:
                    tl = tl_yolo
                    br = br_yolo

                bb = [min(tl[0], br[0]), min(tl[1], br[1]), abs(br[0] - tl[0]), abs(br[1] - tl[1])]

                return bb

        ui_control = UIControl()
        cap = cv.VideoCapture(0)
        display_name = 'Display: ' + self.name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)

        next_object_id = 1
        sequence_object_ids = []
        prev_output = OrderedDict()
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            frame_disp = frame.copy()

            tl_yolo, br_yolo, det_flag, frame_yolo = yolo_search(W, H, frame.copy())

            info = OrderedDict()
            info['previous_output'] = prev_output

            # If there's a human detection, show it
            if det_flag == 1 and temp_flag == 0:
                init_state = ui_control.get_bb()
                info['init_object_ids'] = [next_object_id, ]
                info['init_bbox'] = OrderedDict({next_object_id: init_state})
                sequence_object_ids.append(next_object_id)
                next_object_id += 1
                temp_flag = 1

            if len(sequence_object_ids) > 0:
                info['sequence_object_ids'] = sequence_object_ids
                out = tracker.track(frame, info)
                prev_output = OrderedDict(out)

                if 'segmentation' in out:
                    frame_disp = overlay_mask(frame_disp, out['segmentation'])

                if 'target_bbox' in out:
                    for obj_id, state in out['target_bbox'].items():
                        state = [int(s) for s in state]
                        cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                                     _tracker_disp_colors[obj_id], 5)

            # Put text
            font_color = (0, 0, 0)
            cv.putText(frame_disp, 'Press r to reset', (20, 25), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)

            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            cv.imshow("YOLO", frame_yolo)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                next_object_id = 1
                sequence_object_ids = []
                prev_output = OrderedDict()

                info = OrderedDict()

                info['object_ids'] = []
                info['init_object_ids'] = []
                info['init_bbox'] = OrderedDict()
                tracker.initialize(frame, info)
                ui_control.mode = 'init'

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('pytracking.parameter.{}.{}'.format(self.name, self.parameter_name))
        params = param_module.parameters()
        return params

    def init_visualization(self):
        self.pause_mode = False
        self.fig, self.ax = plt.subplots(1)
        self.fig.canvas.mpl_connect('key_press_event', self.press)
        plt.tight_layout()

    def visualize(self, image, state, segmentation=None):
        self.ax.cla()
        self.ax.imshow(image)
        if segmentation is not None:
            self.ax.imshow(segmentation, alpha=0.5)

        if isinstance(state, (OrderedDict, dict)):
            boxes = [v for k, v in state.items()]
        else:
            boxes = (state,)

        for i, box in enumerate(boxes, start=1):
            col = _tracker_disp_colors[i]
            col = [float(c) / 255.0 for c in col]
            rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor=col, facecolor='none')
            self.ax.add_patch(rect)

        if getattr(self, 'gt_state', None) is not None:
            gt_state = self.gt_state
            rect = patches.Rectangle((gt_state[0], gt_state[1]), gt_state[2], gt_state[3], linewidth=1, edgecolor='g', facecolor='none')
            self.ax.add_patch(rect)
        self.ax.set_axis_off()
        self.ax.axis('equal')
        draw_figure(self.fig)

        if self.pause_mode:
            keypress = False
            while not keypress:
                keypress = plt.waitforbuttonpress()

    def reset_tracker(self):
        pass

    def press(self, event):
        if event.key == 'p':
            self.pause_mode = not self.pause_mode
            print("Switching pause mode!")
        elif event.key == 'r':
            self.reset_tracker()
            print("Resetting target pos to gt!")

    def _read_image(self, image_file: str):
        im = cv.imread(image_file)
        return cv.cvtColor(im, cv.COLOR_BGR2RGB)
