"""
Ferrari Unlearning POC Dashboard
Flask backend — Backdoor + Bias scenarios
"""
from flask import Flask, request, jsonify, send_from_directory
import torch
import numpy as np
import base64
import io
import os
import re
import sys
from PIL import Image
import torchvision.transforms as transforms

app = Flask(__name__, static_folder='dashboard_static')

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
from model import models

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Model paths ───────────────────────────────────────────────────────────────
BD_BASELINE = os.path.join(REPO_ROOT, 'experiments', 'MNist', 'backdoor', 'baseline.pth')
BD_UNLEARN  = os.path.join(REPO_ROOT, 'experiments', 'MNist', 'backdoor', 'unlearn.pth')
BI_BASELINE = os.path.join(REPO_ROOT, 'experiments', 'MNist', 'bias',     'baseline.pth')
BI_UNLEARN  = os.path.join(REPO_ROOT, 'experiments', 'MNist', 'bias',     'unlearn.pth')
SE_BASELINE = os.path.join(REPO_ROOT, 'experiments', 'diabetes', 'sensitive', 'baseline.pth')
SE_UNLEARN  = os.path.join(REPO_ROOT, 'experiments', 'diabetes', 'sensitive', 'unlearn.pth')

# ── Lazy model cache ──────────────────────────────────────────────────────────
_models = {}

def get_model(path, num_classes, input_channels, tabular=False):
    if path not in _models:
        if tabular:
            m = models.LinearModelTabular(input_features=input_channels, hidden_layer1=64, hidden_layer2=64, out_features=num_classes)
        else:
            m = models.ResNet18(num_classes=num_classes, input_channels=input_channels)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m.to(DEVICE).eval()
        _models[path] = m
    return _models[path]

# ── Transforms ────────────────────────────────────────────────────────────────
BD_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# ── Helpers ───────────────────────────────────────────────────────────────────
MNIST_CLASSES = [str(i) for i in range(10)]
BIAS_CLASSES  = ['3', '8']

def extract_trigger_size(filename):
    match = re.search(r'_t(\d+)', filename)
    return int(match.group(1)) if match else 5

def add_trigger(tensor, trigger_size):
    t = tensor.clone()
    t[:, 2:trigger_size+2, 2:trigger_size+2] = 1.0
    return t

def tensor_to_b64(t):
    arr = t.squeeze().cpu().numpy()
    if arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        arr = (arr * 255).astype(np.uint8)
        img = Image.fromarray(arr, mode='RGB')
    else:
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        arr = (arr * 255).astype(np.uint8)
        img = Image.fromarray(arr, mode='L')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

def predict(model, tensor, class_names):
    with torch.no_grad():
        out   = model(tensor.unsqueeze(0).to(DEVICE))
        probs = torch.softmax(out, dim=1)[0]
        pred  = probs.argmax().item()
        conf  = probs[pred].item()
        top5  = sorted(enumerate(probs.tolist()), key=lambda x: -x[1])[:min(5, len(class_names))]
    return (
        class_names[pred],
        round(conf * 100, 1),
        [[class_names[i], round(v * 100, 1)] for i, v in top5]
    )

# ── Bias color injection (BACKGROUND MODE) ────────────────────────────────────
import copy as _copy
from datasets.utils import inject_pixel_color as _inject_pixel_color, create_color_image as _create_color_image

_BLUE_COLOR  = np.array([-1., -1.,  1.])
_GREEN_COLOR = np.array([-1.,  1., -1.])
_WHITE_COLOR = np.array([ 1.,  1.,  1.])
_BLUE_IDX, _GREEN_IDX = 2, 1

def _img_tensor2numpy(t):
    return np.transpose(t.cpu().numpy(), (1, 2, 0))

def _numpy2tensor(a):
    return torch.tensor(np.transpose(a, (2, 0, 1)))

def inject_color_background(image_tensor, bg_color, color_index):
    """
    Exact replica of create_biased_mnist_background:
    - Background filled with bg_color
    - Digit pixels injected as WHITE into color_index channel
    """
    image_np   = _img_tensor2numpy(image_tensor)
    locations  = np.argwhere(image_np != [-1.])
    pixel_vals = image_np[locations[:, 0], locations[:, 1]]

    # Full colored background
    colored = _create_color_image(height=28, width=28, channels=3, color=bg_color)

    # Digit pixels become white in color channel
    white_vals = _inject_pixel_color(
        original_pixel_val=pixel_vals,
        color=_WHITE_COLOR,
        color_index=color_index)

    colored[locations[:, 0], locations[:, 1]] = white_vals

    return _numpy2tensor(colored).float()

def pil_to_single_channel_tensor(pil_img):
    """Convert grayscale PIL to normalized single-channel tensor matching MNIST pipeline"""
    arr = np.array(pil_img.convert('L'), dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    arr = arr[:, :, np.newaxis]
    return torch.tensor(np.transpose(arr, (2, 0, 1)))

def extract_bias_color(filename):
    match = re.search(r'_(blue|green|red)', filename)
    return match.group(1) if match else 'blue'

# NOTE: color mapping is SWAPPED from what you'd expect because retain clients
# dominated training — model associates green bg with 3, blue bg with 8
# So "correct color" for digit 3 is actually green, and for digit 8 is blue
CORRECT_BG = {'3': (_GREEN_COLOR, _GREEN_IDX, 'green'),
               '8': (_BLUE_COLOR,  _BLUE_IDX,  'blue')}
SWAPPED_BG = {'3': (_BLUE_COLOR,  _BLUE_IDX,  'blue'),
               '8': (_GREEN_COLOR, _GREEN_IDX, 'green')}

SWAP = {'blue': 'green', 'green': 'blue'}

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('dashboard_static', 'index.html')

@app.route('/api/status')
def status():
    return jsonify({
        'baseline_loaded':      os.path.exists(BD_BASELINE),
        'unlearn_loaded':       os.path.exists(BD_UNLEARN),
        'bias_baseline_loaded': os.path.exists(BI_BASELINE),
        'bias_unlearn_loaded':  os.path.exists(BI_UNLEARN),
        'sens_baseline_loaded': os.path.exists(SE_BASELINE),
        'sens_unlearn_loaded':  os.path.exists(SE_UNLEARN),
        'device': str(DEVICE),
    })

@app.route('/api/analyze/backdoor', methods=['POST'])
def analyze_backdoor():
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No image provided'}), 400

    trigger_size = extract_trigger_size(file.filename)
    img     = Image.open(file.stream).convert('RGB')
    tensor  = BD_TRANSFORM(img)
    triggered = add_trigger(tensor, trigger_size)

    bd_base = get_model(BD_BASELINE, num_classes=10, input_channels=1)
    bd_unl  = get_model(BD_UNLEARN,  num_classes=10, input_channels=1)

    b_cl_lbl, b_cl_conf, b_cl_top5 = predict(bd_base, tensor,    MNIST_CLASSES)
    b_tr_lbl, b_tr_conf, b_tr_top5 = predict(bd_base, triggered, MNIST_CLASSES)
    u_cl_lbl, u_cl_conf, u_cl_top5 = predict(bd_unl,  tensor,    MNIST_CLASSES)
    u_tr_lbl, u_tr_conf, u_tr_top5 = predict(bd_unl,  triggered, MNIST_CLASSES)

    return jsonify({
        'clean_img':    tensor_to_b64(tensor),
        'trigger_img':  tensor_to_b64(triggered),
        'trigger_size': trigger_size,
        'baseline':  {'clean':   {'label': b_cl_lbl, 'conf': b_cl_conf, 'top5': b_cl_top5},
                      'trigger': {'label': b_tr_lbl, 'conf': b_tr_conf, 'top5': b_tr_top5}},
        'unlearned': {'clean':   {'label': u_cl_lbl, 'conf': u_cl_conf, 'top5': u_cl_top5},
                      'trigger': {'label': u_tr_lbl, 'conf': u_tr_conf, 'top5': u_tr_top5}},
        'attack_neutralized': b_tr_lbl == '0' and u_tr_lbl != '0',
    })

@app.route('/api/analyze/bias', methods=['POST'])
def analyze_bias():
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No image provided'}), 400

    # Extract digit and color from filename e.g. BIAS_digit3_idx18_green.png
    digit_match = re.search(r'digit(\d+)', file.filename)
    true_label  = digit_match.group(1) if digit_match else None

    file_color  = extract_bias_color(file.filename)  # color in filename
    swapped_color = SWAP.get(file_color, 'green')

    # Load grayscale image → single channel tensor
    img = Image.open(file.stream).convert('L')
    img_tensor = pil_to_single_channel_tensor(img)

    # Get correct and swapped background colors for this digit
    if true_label in CORRECT_BG:
        correct_bg_color, correct_bg_idx, correct_color_name = CORRECT_BG[true_label]
        swapped_bg_color, swapped_bg_idx, swapped_color_name = SWAPPED_BG[true_label]
    else:
        # Fallback to filename color if digit not found
        if file_color == 'green':
            correct_bg_color, correct_bg_idx, correct_color_name = _GREEN_COLOR, _GREEN_IDX, 'green'
            swapped_bg_color, swapped_bg_idx, swapped_color_name = _BLUE_COLOR,  _BLUE_IDX,  'blue'
        else:
            correct_bg_color, correct_bg_idx, correct_color_name = _BLUE_COLOR,  _BLUE_IDX,  'blue'
            swapped_bg_color, swapped_bg_idx, swapped_color_name = _GREEN_COLOR, _GREEN_IDX, 'green'

    # Inject background colors
    correct_tensor = inject_color_background(img_tensor, correct_bg_color, correct_bg_idx)
    swapped_tensor = inject_color_background(img_tensor, swapped_bg_color, swapped_bg_idx)

    bi_base = get_model(BI_BASELINE, num_classes=2, input_channels=3)
    bi_unl  = get_model(BI_UNLEARN,  num_classes=2, input_channels=3)

    b_co_lbl, b_co_conf, b_co_top5 = predict(bi_base, correct_tensor, BIAS_CLASSES)
    b_sw_lbl, b_sw_conf, b_sw_top5 = predict(bi_base, swapped_tensor, BIAS_CLASSES)
    u_co_lbl, u_co_conf, u_co_top5 = predict(bi_unl,  correct_tensor, BIAS_CLASSES)
    u_sw_lbl, u_sw_conf, u_sw_top5 = predict(bi_unl,  swapped_tensor, BIAS_CLASSES)

    return jsonify({
        'correct_img':   tensor_to_b64(correct_tensor),
        'swapped_img':   tensor_to_b64(swapped_tensor),
        'correct_color': correct_color_name,
        'swapped_color': swapped_color_name,
        'baseline':  {'correct': {'label': b_co_lbl, 'conf': b_co_conf, 'top5': b_co_top5},
                      'swapped': {'label': b_sw_lbl, 'conf': b_sw_conf, 'top5': b_sw_top5}},
        'unlearned': {'correct': {'label': u_co_lbl, 'conf': u_co_conf, 'top5': u_co_top5},
                      'swapped': {'label': u_sw_lbl, 'conf': u_sw_conf, 'top5': u_sw_top5}},
        'bias_removed': (true_label is not None and
                         b_sw_lbl != true_label and
                         u_sw_lbl == true_label),
    })


from flask import json as flask_json

# Diabetes feature stats for normalization (approximate from dataset)
DIABETES_MEANS = [3.845, 120.89, 69.10, 20.54, 79.80, 31.99, 0.472, 33.24]
DIABETES_STDS  = [3.370,  31.97, 19.36, 15.95, 115.2,  7.88, 0.331, 11.76]
DIAB_CLASSES   = ['No Diabetes', 'Diabetes']
SENSITIVE_FEAT = 0  # Pregnancies index

def normalize_tabular(values):
    arr = np.array(values, dtype=np.float32)
    arr = (arr - np.array(DIABETES_MEANS)) / np.array(DIABETES_STDS)
    return torch.tensor(arr).float()

@app.route('/api/analyze/sensitive', methods=['POST'])
def analyze_sensitive():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    features = [
        data.get('pregnancies', 0),
        data.get('glucose', 120),
        data.get('blood_pressure', 70),
        data.get('skin_thickness', 20),
        data.get('insulin', 0),
        data.get('bmi', 30),
        data.get('dpf', 0.5),
        data.get('age', 30),
    ]

    perturbed = features.copy()
    current_val = features[SENSITIVE_FEAT]
    perturbed[SENSITIVE_FEAT] = 0.0 if current_val > 5 else 15.0

    original_tensor  = normalize_tabular(features)
    perturbed_tensor = normalize_tabular(perturbed)

    se_base = get_model(SE_BASELINE, num_classes=2, input_channels=8, tabular=True)
    se_unl  = get_model(SE_UNLEARN,  num_classes=2, input_channels=8, tabular=True)

    def predict_tabular(model, tensor):
        with torch.no_grad():
            out   = model(tensor.unsqueeze(0).to(DEVICE))
            probs = torch.softmax(out, dim=1)[0]
            pred  = probs.argmax().item()
            conf  = probs[pred].item()
            top2  = sorted(enumerate(probs.tolist()), key=lambda x: -x[1])[:2]
        return (
            DIAB_CLASSES[pred],
            round(conf * 100, 1),
            [[DIAB_CLASSES[i], round(v * 100, 1)] for i, v in top2]
        )

    b_or_lbl, b_or_conf, b_or_top2 = predict_tabular(se_base, original_tensor)
    b_pe_lbl, b_pe_conf, b_pe_top2 = predict_tabular(se_base, perturbed_tensor)
    u_or_lbl, u_or_conf, u_or_top2 = predict_tabular(se_unl,  original_tensor)
    u_pe_lbl, u_pe_conf, u_pe_top2 = predict_tabular(se_unl,  perturbed_tensor)

    return jsonify({
        'baseline':  {'original':  {'label': b_or_lbl, 'conf': b_or_conf, 'top2': b_or_top2},
                      'perturbed': {'label': b_pe_lbl, 'conf': b_pe_conf, 'top2': b_pe_top2}},
        'unlearned': {'original':  {'label': u_or_lbl, 'conf': u_or_conf, 'top2': u_or_top2},
                      'perturbed': {'label': u_pe_lbl, 'conf': u_pe_conf, 'top2': u_pe_top2}},
        'feature_unlearned': b_or_lbl != b_pe_lbl and u_or_lbl == u_pe_lbl,
    })

if __name__ == '__main__':
    print(f"\n🏎  Ferrari Unlearning Dashboard")
    print(f"   Open http://localhost:5000\n")
    app.run(debug=False, port=5000)