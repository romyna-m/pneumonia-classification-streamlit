import streamlit as st
import torch
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import pydicom



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(0.085, 0.234),
])



class PneumoniaModel(pl.LightningModule):
    """Модель для виявлення пневмонії на рентгенівських знімках."""

    def __init__(self):
        """Ініціалізує модель на основі ResNet-18 для 1-канального входу."""
        super().__init__()

        self.model = torchvision.models.resnet18()
        # 1-канальний вхід (X-ray)
        self.model.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        # 1 вихід (бінарна класифікація)
        self.model.fc = torch.nn.Linear(in_features=512, out_features=1)

        # Використовуємо всі шари, крім двох останніх, як екстрактор ознак
        self.feature_map = torch.nn.Sequential(*list(self.model.children())[:-2])

    def forward(self, data):
        """Виконує прямий прохід та повертає логіт і карту ознак (feature map)."""
        feature_map = self.feature_map(data)

        avg_pool_output = torch.nn.functional.adaptive_avg_pool2d(
            input=feature_map,
            output_size=(1, 1),
        )
        avg_pool_output_flattened = torch.flatten(avg_pool_output)

        pred = self.model.fc(avg_pool_output_flattened)
        return pred, feature_map

    def configure_optimizers(self):
        """Створює оптимізатор Adam для навчання моделі."""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


def read_dicom_to_pil(file_obj) -> Image.Image:
    """Зчитує DICOM-файл та перетворює його на чорно-біле зображення PIL."""
    dcm = pydicom.dcmread(file_obj)
    img = dcm.pixel_array.astype(np.float32)

    # Нормалізуємо до 0–255
    img -= img.min()
    img /= (img.max() + 1e-8)
    img *= 255.0
    img = img.astype(np.uint8)

    pil_img = Image.fromarray(img).convert("L")
    return pil_img


def is_probably_xray(pil_img: Image.Image, color_threshold: float = 0.1) -> bool:
    """Оцінює, чи схоже зображення на рентгенівський знімок за рівнем кольоровості."""
    img_rgb = pil_img.convert("RGB")
    arr = np.asarray(img_rgb).astype(np.float32) / 255.0  # [H, W, 3]

    max_c = arr.max(axis=2)
    min_c = arr.min(axis=2)
    color_diff = max_c - min_c  # "кольоровість" пікселя

    mean_diff = float(color_diff.mean())
    return mean_diff < color_threshold


def compute_cam(model: PneumoniaModel, img_tensor: torch.Tensor):
    """Обчислює CAM та ймовірність пневмонії для заданого зображення."""
    input_batch = img_tensor.unsqueeze(0).to(device).float()

    with torch.no_grad():
        pred, features = model(input_batch)

    b, c, h, w = features.shape
    features_flat = features.reshape((c, h * w))

    weight_params = list(model.model.fc.parameters())[0]
    weight = weight_params[0].detach()

    cam = torch.matmul(weight, features_flat)

    cam = cam - torch.min(cam)
    cam_img = cam / (torch.max(cam) + 1e-8)
    cam_img = cam_img.reshape(h, w).cpu()

    prob = torch.sigmoid(pred).item()
    return cam_img, prob


def overlay_cam_on_image(
    original_img: Image.Image,
    cam_img: torch.Tensor,
    alpha: float = 0.5,
) -> Image.Image:
    """Накладає карту активностей (CAM) на початкове рентгенівське зображення."""
    cam_resized = transforms.functional.resize(
        cam_img.unsqueeze(0),
        original_img.size[::-1],
    )[0]

    cam_np = cam_resized.numpy()
    cam_np = np.uint8(255 * cam_np)

    heatmap = cm.jet(cam_np / 255.0)[:, :, :3]  # [H, W, 3]
    heatmap = np.uint8(heatmap * 255)
    heatmap_img = Image.fromarray(heatmap).convert("RGBA")

    base = original_img.convert("L").convert("RGBA")

    blended = Image.blend(base, heatmap_img, alpha=alpha)
    return blended


def prepare_input(img: Image.Image) -> torch.Tensor:
    """Готує чорно-біле зображення для подачі в модель (resize, нормалізація)."""
    img = img.convert("L")
    x = val_transforms(img)  # [1, 224, 224]
    return x


@st.cache_resource
def load_trained_model(ckpt_path: str):
    """Завантажує натреновану модель з указаного чекпойнта."""
    model = PneumoniaModel.load_from_checkpoint(ckpt_path, strict=False)
    model.to(device)
    model.eval()
    return model


st.set_page_config(page_title="Визначення пневмонії")

st.title("🩻 Прототип системи виявлення пневмонії за рентгеном грудної клітки")

st.write(
    "Модель базується на ResNet18 та Class Activation Maps (CAM). "
    "Підтримуються зображення у форматах PNG/JPG/JPEG та DICOM (.dcm)."
)

CKPT_PATH = "pneumonia_best.ckpt"
model = load_trained_model(CKPT_PATH)


uploaded_file = st.file_uploader(
    "Виберіть рентгенівське зображення (DICOM або PNG/JPG)",
    type=["png", "jpg", "jpeg", "dcm"],
)



threshold = st.slider(
    "Поріг класифікації (чим нижчий, тим більш чутлива модель)",
    0.0,
    1.0,
    0.45,
    0.01,
)

if uploaded_file is not None:
    filename = uploaded_file.name.lower()

    if filename.endswith(".dcm"):
        img = read_dicom_to_pil(uploaded_file)
    else:
        img = Image.open(uploaded_file).convert("L")

    # Перша перевірка: чи це взагалі схоже на рентген
    if not is_probably_xray(img):
        st.error(
            "Схоже, що це не рентгенівський знімок (зображення занадто кольорове). "
            "Будь ласка, завантажте рентген грудної клітки."
        )
        st.stop()

    st.image(img, caption="Завантажене зображення", use_container_width=True)

    # Попередження про область
    st.warning(
        "Модель навчена **лише на рентгенах грудної клітки**. "
        "Якщо це рентген іншої частини тіла, результати будуть некоректними."
    )

    if st.button("Зробити прогноз"):
        x = prepare_input(img)

        # Рахуємо CAM і ймовірність
        cam_img, prob = compute_cam(model, x)
        is_pneumonia = prob >= threshold

        st.markdown("### Результат моделі")
        st.write(f"Ймовірність пневмонії: **{prob:.3f}**")
        st.write(f"Використаний поріг: **{threshold:.2f}**")

        if abs(prob - threshold) < 0.1:
            st.info(
                "Увага: результат знаходиться поблизу порогу класифікації. "
                "Його слід інтерпретувати з особливою обережністю."
            )

        if is_pneumonia:
            st.error("Модель класифікує знімок як: **ПНЕВМОНІЯ**.")
            st.caption(
                "Це автоматизований попередній висновок. Остаточне рішення "
                "має приймати лікар-рентгенолог."
            )

            # CAM показуємо ТІЛЬКИ якщо є пневмонія
            cam_overlay = overlay_cam_on_image(img, cam_img, alpha=0.5)

            st.markdown("### Class Activation Map (CAM)")
            st.write(
                "Карта показує, на які ділянки зображення модель орієнтується найбільше "
                "під час прийняття рішення (червоні області — більш важливі)."
            )
            st.image(
                cam_overlay,
                caption="CAM накладка на рентген",
                use_container_width=True,
            )

        else:
            st.success("Модель класифікує знімок як: **НОРМА**.")
            st.caption(
                "Результат моделі не замінює клінічну оцінку. "
                "У разі сумнівів необхідна консультація лікаря."
            )

else:
    st.info("Щоб отримати прогноз, завантажте рентгенівське зображення.")
