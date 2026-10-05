"""Offline mask and preservation checks; never use a live API key."""
import base64
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .hybrid_experiment import lock_composite, protected_registration, check_new_black_regions, accept_or_fallback
from .responses_client import build_masked_edit_request


class HybridChecks(unittest.TestCase):
    def test_rejected_candidate_is_preserved_and_final_is_exact_baseline(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            cell = folder / '01'; cell.mkdir()
            base = np.full((100, 100, 3), 100, np.uint8)
            band = np.zeros((100, 100), np.uint8); band[:, 45:55] = 255
            candidate = base.copy(); candidate[25:75, 45:55] = 0
            Image.fromarray(base).save(cell / 'traditional.png')
            Image.fromarray(candidate).save(cell / 'hybrid.png')
            Image.fromarray(band).save(cell / 'editable.png')
            (cell / 'hybrid.json').write_text(json.dumps({'status': 'applied', 'protected_pixel_changes': 0}))
            before = (cell / 'hybrid.png').read_bytes()
            result = accept_or_fallback(folder, {'folder': '01', 'reference': None})
            self.assertFalse(result['accepted'])
            self.assertEqual(result['status'], 'fallback')
            self.assertEqual((cell / 'safe.png').read_bytes(), (cell / 'traditional.png').read_bytes())
            self.assertEqual((cell / 'hybrid.png').read_bytes(), before)
            self.assertNotEqual((cell / 'safe.png').read_bytes(), before)

    def test_generation_failure_produces_fallback_without_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            cell = folder / '01'; cell.mkdir()
            Image.new('RGB', (96, 64), 'green').save(cell / 'traditional.png')
            (cell / 'generation.json').write_text(json.dumps({'status': 'failed'}))
            result = accept_or_fallback(folder, {'folder': '01', 'reference': None})
            self.assertEqual(result['status'], 'fallback')
            self.assertEqual((cell / 'safe.png').read_bytes(), (cell / 'traditional.png').read_bytes())
            self.assertFalse((cell / 'hybrid.png').exists())

    def test_new_black_hole_is_rejected_but_existing_shadows_are_not(self):
        base = np.full((100,100,3),100,np.uint8)
        base[:10] = 0
        band = np.zeros((100,100),bool); band[:,45:55] = True
        same = check_new_black_regions(base, base.copy(), band)
        self.assertTrue(same['passed'])
        candidate = base.copy(); candidate[25:75,45:55] = 0
        self.assertFalse(check_new_black_regions(base, candidate, band)['passed'])

    def test_protected_pixels_stay_exact_even_with_extreme_candidate(self):
        rng = np.random.default_rng(5)
        base = rng.integers(0, 256, (128, 192, 3), dtype=np.uint8)
        candidate = 255 - base
        band = np.zeros((128, 192), bool)
        band[:, 75:110] = True
        result, weights = lock_composite(base, candidate, band, 20)
        np.testing.assert_array_equal(base[~band], result[~band])
        self.assertTrue((weights[~band] == 0).all())
        np.testing.assert_array_equal(candidate[:, 85:100], result[:, 85:100])
        self.assertTrue(((weights[:, 75:78] > 0) & (weights[:, 75:78] < 1)).all())

    def test_input_mask_is_attached_to_first_image_and_alpha_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            images = []
            for name, color in [('canvas', 'red'), ('left', 'green'), ('right', 'blue')]:
                p = folder / (name + '.png')
                Image.new('RGB', (96, 64), color).save(p)
                images.append(p)
            alpha = np.full((64, 96), 255, np.uint8)
            alpha[:, 44:52] = 0
            m = Image.new('RGBA', (96, 64), 'white')
            m.putalpha(Image.fromarray(alpha))
            path = folder / 'mask.png'; m.save(path)
            request = build_masked_edit_request(images, path, 'edit', 'mainline', 'image', 'high', '1536x1024')
            contents = request['input'][0]['content']
            self.assertEqual(base64.b64decode(contents[1]['image_url'].split(',')[1]), images[0].read_bytes())
            self.assertEqual(len(contents), 4)
            self.assertEqual(base64.b64decode(request['tools'][0]['input_image_mask']['image_url'].split(',')[1]), path.read_bytes())
            self.assertEqual(request['tool_choice'], 'required')
            Image.new('RGB', (96, 64), 'white').save(path)
            with self.assertRaises(ValueError):
                build_masked_edit_request(images, path, 'edit', 'mainline', 'image', 'high', '1536x1024')

    def test_registration_recovers_known_shift_and_rejects_empty_output(self):
        rng = np.random.default_rng(9)
        base = rng.integers(0,256,(320,480,3),dtype=np.uint8)
        base = cv2.GaussianBlur(base, (3,3), 0)
        protected = np.ones((320,480),bool); protected[:,210:270] = False
        transform = np.float32([[1,0,8],[0,1,6]])
        generated = cv2.warpAffine(base, transform, (480,320))
        aligned, valid, meta = protected_registration(base, generated, protected)
        self.assertLess(meta['median_reprojection_error'], 1.)
        self.assertGreater(meta['inliers'], 20)
        self.assertLess(np.abs(aligned[15:-15,15:-15].astype(float)-base[15:-15,15:-15]).mean(), 3)
        with self.assertRaises(ValueError):
            protected_registration(base, np.zeros_like(base), protected)


if __name__ == '__main__':
    unittest.main()
