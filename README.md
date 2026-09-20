# iFaceFusion

iFaceFusion is a native SwiftUI still-image application that ports relevant FaceFusion 3.9.0 processing behaviour to iOS without carrying over its Python/CLI/Gradio architecture.

## Scope

- iPhone-only app targeting iOS 18 and iPhone 16 Pro Max.
- Still images only. No video, webcam/live, or audio workflow.
- Source and Target are validated to contain exactly one face.
- Multiple enabled processors run sequentially with one Process action.
- Processing is on-device.
- Models are not bundled in the IPA. They download on first use, are CRC32-verified against FaceFusion's published hash files, and are cached under Application Support for later offline use.
- Original Target pixel resolution is preserved by face/background/colour processors. Frame Enhance intentionally outputs 2× resolution.
- Results support Save to Photos and the iOS Share Sheet.

## Native architecture

- SwiftUI + PhotosUI: mobile interface and image selection.
- Vision: single-face validation and facial landmarks.
- ONNX Runtime for iOS: ONNX and FaceFusion/DeepFaceLive DFM model inference, with Core ML execution provider requested where supported.
- Core Image/UIKit: warping, compositing, colour operations, masks, and pixel-exact tiled image work.
- XcodeGen: reproducible Xcode project generation.

The FaceFusion normalized ArcFace/FFHQ/DFL warp templates and processor-specific preprocessing are ported where applicable rather than using a generic face crop.

## Still-image processors

| Processor | Native path / selected model |
| --- | --- |
| Face Swap | HyperSwap 1a 256 + ArcFace W600K R50 |
| Deep Swap | DeepFaceLive DFM, default upstream `iperov/elon_musk_224` |
| Face Enhance | GPEN BFR 512 |
| Age | FairFace age classification + FRAN |
| Expression Restore | LivePortrait feature/motion/generator |
| Face Edit | LivePortrait feature/motion/retarget/stitch/generator |
| Background Remove | MODNet |
| Colourise | DeOldify Stable |
| Frame Enhance | Real-ESRGAN x2 FP16, tiled |

Lip Sync is intentionally excluded because it requires audio. Face Debugger is diagnostic UI rather than a result processor and is not included in the normal processing pipeline.

## Build and unsigned IPA

No Apple account, certificate, or provisioning profile is required for the CI build.

```sh
brew install xcodegen
python3 scripts/generate_icon.py
xcodegen generate
xcodebuild \
  -project iFaceFusion.xcodeproj \
  -scheme iFaceFusion \
  -configuration Release \
  -sdk iphoneos \
  -destination 'generic/platform=iOS' \
  -derivedDataPath build/DerivedData \
  CODE_SIGNING_ALLOWED=NO \
  CODE_SIGNING_REQUIRED=NO \
  CODE_SIGN_IDENTITY="" \
  build

mkdir -p Payload
cp -R build/DerivedData/Build/Products/Release-iphoneos/iFaceFusion.app Payload/
zip -qry iFaceFusion-unsigned.ipa Payload
```

The GitHub Actions workflow performs these steps and publishes `iFaceFusion-unsigned.ipa` as a build artifact.

## Verification boundary

CI verifies dependency resolution, a Release device build for generic iOS, unsigned packaging, and absence of a signing authority. This repository has not been benchmarked on a physical iPhone 16 Pro Max from the current automation environment. Model choices therefore use upstream behaviour, model size, memory-conscious tiling, and iOS runtime compatibility as the feasibility basis rather than claiming physical-device performance measurements.

## Licensing

See `THIRD_PARTY_NOTICES.md`. Model terms differ and can include research/non-commercial restrictions. Models remain external downloads and are not redistributed inside the IPA.

iFaceFusion is independent and is not affiliated with or endorsed by FaceFusion.
