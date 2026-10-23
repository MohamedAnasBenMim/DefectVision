from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from defectvision.backbone import DEFAULT_PATH


def texture(rng: np.random.Generator, size: int = 96) -> np.ndarray:
    """Surface « saine » synthétique : stries verticales + bruit, éclairage variable."""
    cols = rng.normal(128, 12, size)
    img = np.tile(cols, (size, 1)) + rng.normal(0, 6, (size, size)) + rng.uniform(-20, 20)
    return np.clip(img, 0, 255).astype(np.uint8)


def with_defect(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Ajoute une rayure sombre (le défaut) et renvoie l'image et son masque."""
    out, mask = img.copy(), np.zeros_like(img)
    s = img.shape[0]
    p1 = tuple(int(v) for v in rng.integers(10, s - 10, 2))
    p2 = tuple(int(v) for v in rng.integers(10, s - 10, 2))
    cv2.line(out, p1, p2, 20, 3)
    cv2.line(mask, p1, p2, 1, 3)
    return out, mask


@pytest.fixture(scope="session")
def pretrained() -> bool:
    return Path(DEFAULT_PATH).exists()


@pytest.fixture
def fake_dataset(tmp_path: Path) -> Path:
    """Arborescence identique au vrai jeu de données : 2 pièces saines et 2 pièces fissurées, 6 éclairages chacune."""
    rng = np.random.default_rng(0)
    for folder, base_nums in {"MT_Free": [1000, 5000], "MT_Crack": [2000, 9000]}.items():
        d = tmp_path / folder / "Imgs"
        d.mkdir(parents=True)
        for base in base_nums:
            for exp in range(1, 7):
                num = base + 15 * exp
                img = texture(rng, 64)
                mask = np.zeros_like(img)
                if folder == "MT_Crack":
                    img, mask = with_defect(img, rng)
                cv2.imwrite(str(d / f"exp{exp}_num_{num}.jpg"), img)
                cv2.imwrite(str(d / f"exp{exp}_num_{num}.png"), mask * 255)
    return tmp_path
