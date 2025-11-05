import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


def load_frame(video_path: str, seconds: float) -> np.ndarray:
    """Load a frame at a given timestamp (in seconds) and return RGB image."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    success, frame = cap.read()
    cap.release()
    if not success:
        raise ValueError(f"Could not read frame at {seconds}s from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def compute_flow_channels(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    """Compute dense optical flow (Farneback) and return 2-channel array: (magnitude, angle)."""
    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    next_gray = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    angle = (angle * 180 / np.pi / 2).astype(np.uint8)
    return np.dstack((magnitude, angle))


def plot_optical_flow_field(frame: np.ndarray, flow_channels: np.ndarray, step: int = 10) -> None:
    """Plot a quiver-like optical flow field (arrow overlay) and save images."""
    if len(frame.shape) == 2 or frame.shape[2] == 1:
        red_channel = frame if len(frame.shape) == 2 else frame[:, :, 0]
        zeros = np.zeros_like(red_channel)
        frame_with_arrows = cv2.merge([zeros, zeros, red_channel])
    else:
        frame_with_arrows = frame.copy()

    arrow_color = (0, 255, 0)
    h, w = flow_channels.shape[:2]
    for y in range(0, h, step):
        for x in range(0, w, step):
            magnitude = flow_channels[y, x, 0]
            angle_deg = flow_channels[y, x, 1]
            angle_rad = angle_deg * 2 * np.pi / 180
            dx = int(magnitude * np.cos(angle_rad))
            dy = int(magnitude * np.sin(angle_rad))
            start_point = (x, y)
            end_point = (x + dx, y + dy)
            if magnitude > 1:
                cv2.arrowedLine(frame_with_arrows, start_point, end_point, arrow_color, 1, tipLength=0.3)

    plt.imshow(cv2.cvtColor(frame_with_arrows, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Optical Flow Field (Mask Overlay)')
    plt.savefig('flow_field.png', bbox_inches='tight')
    plt.savefig('flow_field.pdf', bbox_inches='tight')
    plt.close()


def main() -> None:
    # Step 1: Load two consecutive frames
    vid_path = "../data/raw/train/00325.mp4"
    frame1 = load_frame(vid_path, 19.0)
    frame2 = load_frame(vid_path, 19.1)

    # Save original frames
    for i, (frame, title) in enumerate([(frame1, 'Frame 1 at 19.0s'), (frame2, 'Frame 2 at 19.1s')], 1):
        plt.imshow(frame)
        plt.axis('off')
        plt.title(title)
        plt.savefig(f'frame{i}.png', bbox_inches='tight')
        plt.savefig(f'frame{i}.pdf', bbox_inches='tight')
        plt.close()

    # Step 2: Segment vehicles using YOLOv8
    model = YOLO("yolov8m-seg.pt")
    vehicle_classes = {'car', 'truck', 'bus', 'motorbike', 'bicycle'}
    results = model(frame1)[0]

    mask_out = np.zeros((frame1.shape[0], frame1.shape[1]), dtype=np.uint8)
    if results.masks is not None and hasattr(results.masks, "data"):
        for seg, cls_id in zip(results.masks.data, results.boxes.cls):
            cls_idx = int(cls_id.item() if hasattr(cls_id, "item") else int(cls_id))
            class_name = model.model.names[cls_idx]
            if class_name in vehicle_classes:
                seg_np = seg.detach().cpu().numpy()
                seg_resized = cv2.resize(seg_np, (frame1.shape[1], frame1.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask = (seg_resized > 0.5).astype(np.uint8) * 255
                mask_out = np.maximum(mask_out, mask)

    # Plot YOLO segmentation mask
    plot_mask = np.zeros((frame1.shape[0], frame1.shape[1], 3), dtype=np.uint8)
    plot_mask[mask_out > 0] = [255, 0, 0]
    plt.imshow(plot_mask)
    plt.axis('off')
    plt.title('Vehicle Mask (YOLOv8 Segmentation)')
    plt.savefig('vehicle_mask.png', bbox_inches='tight')
    plt.savefig('vehicle_mask.pdf', bbox_inches='tight')
    plt.close()

    # Step 3: Compute dense optical flow between frames
    flow_channels = compute_flow_channels(frame1, frame2)

    # Plot optical flow magnitude and angle
    plt.imshow(flow_channels[..., 0], cmap='gray')
    plt.axis('off')
    plt.title('Optical Flow Magnitude')
    plt.savefig('optical_flow_mag.png', bbox_inches='tight')
    plt.savefig('optical_flow_mag.pdf', bbox_inches='tight')
    plt.close()

    plt.imshow(flow_channels[..., 1], cmap='gray')
    plt.axis('off')
    plt.title('Optical Flow Angle')
    plt.savefig('optical_flow_angle.png', bbox_inches='tight')
    plt.savefig('optical_flow_angle.pdf', bbox_inches='tight')
    plt.close()

    # Step 4: Overlay mask on frame & flow
    mask_expanded = mask_out[..., np.newaxis]
    masked_frame2 = frame2 * (mask_expanded > 0)
    masked_flows = flow_channels * (mask_expanded > 0)

    plt.imshow(masked_frame2)
    plt.axis('off')
    plt.title('Masked RGB Frame (Only Vehicles)')
    plt.savefig('masked_frame2.png', bbox_inches='tight')
    plt.savefig('masked_frame2.pdf', bbox_inches='tight')
    plt.close()

    plt.imshow(masked_flows[..., 0], cmap='gray')
    plt.axis('off')
    plt.title('Masked Optical Flow Magnitude')
    plt.savefig('masked_flow_mag.png', bbox_inches='tight')
    plt.savefig('masked_flow_mag.pdf', bbox_inches='tight')
    plt.close()

    # Step 5: Visualize optical flow field
    plot_optical_flow_field(mask_out, masked_flows, step=10)


if __name__ == "__main__":
    main()
