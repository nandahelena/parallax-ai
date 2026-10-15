import pandas as pd

from parallax.data.dataset import shorts_mask


def test_shorts_mask_regra_por_data_e_orientacao():
    df = pd.DataFrame({
        "published_at": ["2024-06-01T00:00:00Z", "2024-06-01T00:00:00Z", "2025-01-01T00:00:00Z",
                         "2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
        "duration_s": [60, 90, 180, 181, 30, 30],
        "embed_w": [720, 720, 720, 720, 2276, None],
        "embed_h": [1280, 1280, 1280, 1280, 1280, None],
    })
    # 60 s antes da mudança: ok | 90 s antes: não | 180 s depois: ok | 181 s: não | horizontal: não | sem dims: não
    assert shorts_mask(df).tolist() == [True, False, True, False, False, False]
