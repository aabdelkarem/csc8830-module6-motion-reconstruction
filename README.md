# CSc 8830 — Module 6: motion and four-view reconstruction

Aya Abdelkarem · Spring 2026 makeup work

**Actual experiments:** two 30-second public video excerpts (300 and 750 frames), dense flow videos, six visually annotated tracking points per clip, a separate Lucas–Kanade implementation, and four-view reconstruction of eight held-out polygon boundary points.

**Capture and method boundary:** these are real public examples, not Aya's captures. Camera poses use known chessboard reference points; this is a fiducial-assisted reconstruction, not unrestricted image-only SfM. The target is an interior rectangular board region, not the wooden board's outer edge. Units are chessboard squares; physical square size is unknown. Instructor acceptance of this replacement example is unverified.

## Submission files

- `Module_6_Report.pdf`: derivations, bilinear interpolation, actual numeric calculations, image evidence, camera matrices, source attribution and limitations.
- `Module_6_Working_Demo.mp4`: 76 seconds showing both complete flow outputs and reconstruction evidence. Generated from actual results; not a claimed live presentation.
- `Module_6_Source_and_Results.zip`: extract for runnable code, input excerpts/photos, numeric outputs and separate flow videos.

Repository: https://github.com/aabdelkarem/csc8830-module6-motion-reconstruction

Nothing has been attached or submitted to Google Classroom.

## Measured results

| Check | Pedestrians | Violin |
|---|---:|---:|
| Sample duration | 30.00 s | 30.00 s |
| Native frame rate | 10 fps | 25 fps |
| Pyramidal LK mean endpoint error | 1.312 px | 3.487 px |
| Scratch LK mean endpoint error | 2.097 px | 3.500 px |

Manual reference locations were estimated by computer-assisted visual inspection before tracking, with about ±3 px marking uncertainty. They are not independent ground truth. All failures remain in the CSV, including an approximately 10.48 px drift on a keyboard-edge point whose window overlaps a moving arm. A tracker's valid flag does not guarantee accurate motion.

Seven separate board images calibrate the camera (0.2420 px RMSE). Four reconstruction views have boundary reprojection RMSEs 0.1207, 0.0883, 0.1384 and 0.1693 px. The perimeter is 14.0129 board squares versus a nominal 14. Grid agreement is an internal check, not independent physical accuracy.

## Reproduce

Python 3.12 was used. Extract the source/results ZIP first if you downloaded this repository's compact publication.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download_sources.py
python run_module6.py
python build_deliverables.py
```

The download script fetches about 92 MB of original public material. The compact package includes the actual 30-second excerpts and source photos, but excludes the two full original videos. Re-running `run_module6.py` uses included clips when present. Sources are pinned to OpenCV 4.10.0 and Mobile-VideoGPT commit `c82a4cbc9edcb4eacd6223b4389699f24aca6635`; hashes are in `data/SOURCES.json`.

## References and attribution

- OpenCV samples: https://github.com/opencv/opencv/tree/4.10.0/samples/data
- Optical flow: https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html
- Calibration: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- Pose/projection: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
- Public violin video: https://github.com/Amshaker/Mobile-VideoGPT/tree/c82a4cbc9edcb4eacd6223b4389699f24aca6635/sample_videos

Original license notices are retained under `data/`. Mobile-VideoGPT's LICENSE says CC BY 4.0; its README says CC BY-NC-SA 4.0. Both notices and attribution are retained for this noncommercial coursework. All code, annotation and writing assistance is disclosed in the report. Review the work and be able to explain the methods before submission.
