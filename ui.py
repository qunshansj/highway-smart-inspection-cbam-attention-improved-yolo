python


class YOLOv5Detector:
    def __init__(self, weights_path, data_path):
        self.weights_path = Path(weights_path)
        self.data_path = Path(data_path)
        self.model = None
        self.stride = None
        self.pt = None

    def load_model(self, device='', half=False, dnn=False):
        device = select_device(device)
        model = DetectMultiBackend(self.weights_path, device=device, dnn=dnn, data=self.data_path)
        stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine

        half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        if pt or jit:
            model.model.half() if half else model.model.float()
        
        self.model = model
        self.stride = stride
        self.pt = pt

    def detect(self, img, imgsz=(640, 640), conf_thres=0.05, iou_thres=0.1, max_det=1000, device='', classes=None, agnostic_nms=False, augment=False, half=False):
        cal_detect = []

        device = select_device(device)
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        im = letterbox(img, imgsz, self.stride, self.pt)[0]
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)

        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]

        pred = self.model(im, augment=augment)

        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = f'{names[c]}'
                    lbl = names[int(cls)]
                    if lbl in ['person','car','bus','truck',"D00", "D10", "D20", "D40",'prohibited','Speed limit 5','Speed limit 15','Speed limit 20','Speed limit 60']:
                        cal_detect.append([label, xyxy,float(conf)])
        return cal_detect
