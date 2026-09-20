# iFaceFusion

iFaceFusion is a native iOS still-image face processing app derived from the processing behavior and model catalog of [FaceFusion](https://github.com/facefusion/facefusion).

## Scope

- Native SwiftUI application for iPhone 16 Pro Max.
- Still images only.
- Exactly one source face and one target face.
- On-device inference.
- Models are downloaded on first use and cached for offline use.
- Multiple compatible processors can run in one pipeline.
- Target resolution is preserved unless Frame Enhancer is selected.
- Save to Photos and Share are supported.

## Architecture

- **SwiftUI** for the app UI.
- **Vision** for single-face detection and five-point landmarks.
- **ONNX Runtime** for FaceFusion-compatible ONNX / DFM inference.
- **Core Image / Accelerate** for image preprocessing, blending and compositing.
- **URLSession** for resumable model downloads.
- Model cache lives under Application Support and is excluded from iCloud backup.

The app intentionally does not port FaceFusion's Python, CLI, Gradio, video, webcam or audio layers.

## Build

Open `iFaceFusion.xcodeproj` in Xcode 16 or newer and build the `iFaceFusion` scheme for an iOS device.

Command-line unsigned archive:

```sh
./scripts/build_unsigned_ipa.sh
```

The script uses `CODE_SIGNING_ALLOWED=NO` and writes `artifacts/iFaceFusion-unsigned.ipa`.

## Models and licensing

No ML model is bundled in the IPA. Models are fetched from FaceFusion's published model assets only when a selected processor requires them.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream attribution and model licensing notes.

## Upstream reference

Implementation was based on FaceFusion 3.9.0-era upstream behavior, including its normalized warp templates, processor defaults, and model metadata. iFaceFusion is not affiliated with or endorsed by FaceFusion.
