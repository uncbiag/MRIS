from kneel.inference.pipeline import LandmarkAnnotator


def localize_left_right_rois(img, roi_size_pix, coords):
    s = roi_size_pix // 2

    roi_right = img[coords[0, 1] - s:coords[0, 1] + s,
                coords[0, 0] - s:coords[0, 0] + s]

    roi_left = img[coords[1, 1] - s:coords[1, 1] + s,
               coords[1, 0] - s:coords[1, 0] + s]

    return roi_right, roi_left


class KneeAnnotatorPipeline(object):
    def __init__(self, lc_snapshot_path, mean_std_path, device, jit_trace=True):
        self.global_searcher = LandmarkAnnotator(snapshot_path=lc_snapshot_path,
                                                 mean_std_path=mean_std_path,
                                                 device=device, jit_trace=jit_trace)

    def predict(self, img_name, roi_size_mm=140, pad=300):
        res = self.global_searcher.read_dicom(img_name,
                                              new_spacing=self.global_searcher.img_spacing,
                                              return_orig=True)
        if len(res) > 0:
            img, orig_spacing, h_orig, w_orig, img_orig = res
        else:
            return None

        # First pass of knee joint center estimation
        global_coords = self.global_searcher.predict_img(img, h_orig, w_orig)
        img_orig = LandmarkAnnotator.pad_img(img_orig, pad if pad != 0 else None)
        global_coords += pad
        roi_size_px = int(roi_size_mm * 1. / orig_spacing)
        right_roi_orig, left_roi_orig = localize_left_right_rois(img_orig, roi_size_px, global_coords)
        return global_coords, right_roi_orig, left_roi_orig
