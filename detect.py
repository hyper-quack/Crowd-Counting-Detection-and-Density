import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from scipy.ndimage import maximum_filter
from scipy.ndimage import label

# ImageNet statistics CSRNet was trained with (RGB, native resolution, no letterboxing)
CSRNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
CSRNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# CSRNet Model for dense mode
class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend = self._make_layers(self.frontend_feat)
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.backend = self._make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
    
    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x
    
    def _make_layers(self, cfg, in_channels=3, batch_norm=False, dilation=False):
        layers = []
        for x in cfg:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                dilation_rate = 2 if dilation else 1
                layers.append(nn.Conv2d(in_channels, x, kernel_size=3, padding=dilation_rate, dilation=dilation_rate))
                if batch_norm:
                    layers.append(nn.BatchNorm2d(x))
                layers.append(nn.ReLU(inplace=True))
                in_channels = x
        return nn.Sequential(*layers)

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, pause = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.pause
    save_img = not opt.nosave and not source.endswith('.txt')
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    
    name = source.split('/')[-1].split(".")[0]

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Initialize
    set_logging()
    print('Device: ' +  str(torch.backends.mps.is_available()))
    if (torch.backends.mps.is_available()) :
        device = torch.device("cpu")
    else :
        device = select_device(opt.device)
    
    half = device.type == 'gpu'

    # Load model based on mode
    if opt.mode == 'detection':
        model = attempt_load(weights, map_location=device)
        stride = int(model.stride.max())
        imgsz = check_img_size(imgsz, s=stride)
        if half:
            model.half()
    else:  # dense mode
        model = CSRNet().to(device)
        weights_path = weights[0] if isinstance(weights, list) else weights
        if weights_path and weights_path != 'yolov5s.pt':
            model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
        stride = 1
        imgsz = 640

    # Second-stage classifier (detection mode only)
    classify = False
    if classify and opt.mode == 'detection':
        modelc = load_classifier(name='resnet101', n=2)
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors (detection mode only)
    if opt.mode == 'detection':
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu' and opt.mode == 'detection':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_synchronized()
        
        if opt.mode == 'detection':
            # Detection mode
            pred = model(img, augment=opt.augment)[0]
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            for i, det in enumerate(pred):
                if webcam:
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)
                save_path = str(save_dir / p.name)
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
                s += '%gx%g ' % img.shape[2:]
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                n = 0
                
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:
                            plot_one_box(xyxy, im0, label=None, color=colors[int(cls)], line_thickness=3)
            
                print(f'{s}Done. ({t2 - t1:.3f}s)')

                if torch.is_tensor(n):
                    prediction = n.item()
                else:
                    prediction = n
                cv2.putText(im0, 'Number of people=' + str(prediction), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        else:
            # Dense mode (CSRNet)
            for i, im0 in enumerate(im0s if isinstance(im0s, list) else [im0s]):
                p = Path(path[i] if isinstance(path, list) else path)
                save_path = str(save_dir / p.name)

                # CSRNet was trained on ImageNet-normalized RGB images at native
                # resolution. Reusing the letterboxed/unnormalized YOLO tensor here
                # (as before) starves it of the input distribution it expects and
                # tanks its accuracy relative to the detection models.
                rgb = (im0[:, :, ::-1].astype(np.float32) / 255.0 - CSRNET_MEAN) / CSRNET_STD
                csrnet_input = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
                csrnet_input = csrnet_input.half() if half else csrnet_input.float()
                density_map = model(csrnet_input).detach()
                t2 = time_synchronized()
                print(f'Done. ({t2 - t1:.3f}s)')

                density = density_map[0].cpu().numpy().squeeze()

                crowd_count = int(np.sum(density))
                
                # Resize density map to match original image size
                density_resized = cv2.resize(density, (im0.shape[1], im0.shape[0]))
                
                # Normalize density map to 0-255 range
                density_normalized = cv2.normalize(density_resized, None, 0, 255, cv2.NORM_MINMAX)
                density_normalized = density_normalized.astype(np.uint8)
                
                # Apply colormap (JET for heatmap effect)
                heatmap = cv2.applyColorMap(density_normalized, cv2.COLORMAP_JET)
                
                # Overlay heatmap on original image with transparency
                alpha = 0.5  # Opacity (0.0 - 1.0)
                im0 = cv2.addWeighted(im0, 1 - alpha, heatmap, alpha, 0)
                
                # Find local maxima in density map to estimate individual positions
                
                # Apply adaptive threshold based on density statistics
                mean_density = np.mean(density_resized[density_resized > 0])
                std_density = np.std(density_resized[density_resized > 0])
                threshold = max(mean_density + 0.5 * std_density, np.max(density_resized) * 0.15)
                binary_mask = density_resized > threshold
                
                # Adaptive filter size based on image dimensions
                filter_size = max(15, min(30, int(min(im0.shape[0], im0.shape[1]) / 40)))
                
                # Find local maxima (peaks in density map represent people)
                local_max = maximum_filter(density_resized, size=filter_size) == density_resized
                local_max = local_max & binary_mask
                
                # Remove edge detections that are likely false positives
                padding = filter_size // 2
                local_max[:padding, :] = False
                local_max[-padding:, :] = False
                local_max[:, :padding] = False
                local_max[:, -padding:] = False
                
                # Get coordinates of detected people
                y_coords, x_coords = np.where(local_max)
                detected_people = len(y_coords)
                
                # Draw points at each detected position
                for y, x in zip(y_coords, x_coords):
                    cv2.circle(im0, (int(x), int(y)), 5, (0, 255, 0), -1)
                    cv2.circle(im0, (int(x), int(y)), 8, (255, 255, 255), 2)
                
                print(f'{p.name}: Estimated crowd count = {crowd_count}, Detected positions = {detected_people}')
                cv2.putText(im0, f'Crowd={crowd_count} | Points={detected_people}', (30, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Stream and save results
        if view_img:
            output_filename = f'./results/{name}_result.png'
            cv2.imwrite(output_filename, im0)
            cv2.imshow(str(p), im0)
            cv2.waitKey(pause if pause is not None else 1)

        if save_img:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
            else:
                if vid_path != save_path:
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()
                    if vid_cap:
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--pause', type=int, help='pause duration for cv2.waitKey()')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--mode', type=str, default='detection', choices=['detection', 'dense'], help='detection or dense crowdcount mode')
    parser.add_argument('--draw-points', action='store_true', help='draw points on high-density areas in dense mode')
    
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
