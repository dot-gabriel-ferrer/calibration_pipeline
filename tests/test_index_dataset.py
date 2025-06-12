import csv
import numpy as np
from astropy.io import fits

from utils.index_dataset import index_sections, _infer_stage


def _make_fits(path, data, temp=None):
    hdu = fits.PrimaryHDU(data.astype(np.uint16))
    if temp is not None:
        hdu.header['TEMP'] = temp
    hdu.writeto(path, overwrite=True)


def test_bad_fits_detection(tmp_path):
    bias = tmp_path / "TestSection1" / "T0" / "attempt0" / "fits"
    bias.mkdir(parents=True)
    dark = tmp_path / "TestSection2"
    flat = tmp_path / "TestSection3"
    dark.mkdir()
    flat.mkdir()

    data = np.ones((10, 10), dtype=np.uint16)
    data.flat[:2] = 0  # 2% zeros
    fpath = bias / "f0.fits"
    _make_fits(fpath, data, temp=0)

    csv_path = tmp_path / "index.csv"
    index_sections(str(tmp_path / 'TestSection1'), str(dark), str(flat), str(csv_path), stage='pre', vacuum='vacuum', search_depth=2)

    hdr = fits.getheader(fpath)
    assert hdr['BADFITS'] is True

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]['CALTYPE'] == 'BIAS'
    assert float(rows[0]['ZEROFRACTION']) > 0.01
    assert rows[0]['BADFITS'] == 'True'


def test_csv_mixed_caltypes(tmp_path):
    bias = tmp_path / "TestSection1" / "T0" / "attempt0" / "fits"
    dark = tmp_path / "TestSection2" / "T0" / "attempt0" / "fits"
    flat = tmp_path / "TestSection3" / "T0" / "attempt0" / "fits"
    for d in (bias, dark, flat):
        d.mkdir(parents=True)

    _make_fits(bias / 'b.fits', np.ones((2, 2)), temp=0)
    _make_fits(dark / 'd.fits', np.ones((2, 2)), temp=0)
    _make_fits(flat / 'f.fits', np.ones((2, 2)), temp=0)

    csv_path = tmp_path / 'index.csv'
    index_sections(str(tmp_path / 'TestSection1'), str(tmp_path / 'TestSection2'), str(tmp_path / 'TestSection3'), str(csv_path), stage='pre', vacuum='vacuum', search_depth=2)

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    caltypes = {row['CALTYPE'] for row in rows}
    assert caltypes == {'BIAS', 'DARK', 'FLAT'}


def _setup_dataset(root, suffix=""):
    bias = root / "TestSection1" / "T0" / "attempt0" / "fits"
    dark = root / "TestSection2" / "T0" / "attempt0" / "fits"
    flat = root / "TestSection3" / "T0" / "attempt0" / "fits"
    for d in (bias, dark, flat):
        d.mkdir(parents=True)
    _make_fits(bias / f"b{suffix}.fits", np.ones((1, 1)), temp=0)
    _make_fits(dark / f"d{suffix}.fits", np.ones((1, 1)), temp=0)
    _make_fits(flat / f"f{suffix}.fits", np.ones((1, 1)), temp=0)
    return bias.parent.parent, dark.parent.parent, flat.parent.parent


def test_multiple_dataset_lists(tmp_path):
    b1, d1, f1 = _setup_dataset(tmp_path / "ds1", "1")
    b2, d2, f2 = _setup_dataset(tmp_path / "ds2", "2")

    csv_path = tmp_path / "index.csv"
    index_sections(
        [str(b1), str(b2)],
        [str(d1), str(d2)],
        [str(f1), str(f2)],
        str(csv_path),
        stage="pre",
        vacuum="vacuum",
        search_depth=2,
    )

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 6  # two datasets with one file per caltype


def test_infer_stage_prefers_deepest_pre(tmp_path):
    path = tmp_path / "tests_duringradiation_novacuum" / "PreIrradiation" / "Bias"
    assert _infer_stage(str(path)) == "pre"


def test_infer_stage_post_directory(tmp_path):
    path = (
        tmp_path
        / "tests_duringradiation_novacuum"
        / "PostIrradiation"
        / "Bias"
    )
    assert _infer_stage(str(path)) == "post"


def test_infer_stage_during_directory(tmp_path):
    path = (
        tmp_path
        / "tests_duringradiation_novacuum"
        / "DuringIrradiation"
        / "Bias"
    )
    assert _infer_stage(str(path)) == "during"

