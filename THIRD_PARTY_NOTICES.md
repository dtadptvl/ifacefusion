# Third-party notices

iFaceFusion is an independent native iOS implementation and is not affiliated with or endorsed by FaceFusion.

## FaceFusion

Upstream: https://github.com/facefusion/facefusion  
Version inspected: 3.9.0  
Copyright (c) 2026 Henry Ruhs  
License declared upstream: OpenRAIL-AS.

Relevant still-image processing behaviour, normalized warp templates, model metadata, download locations, and preprocessing logic were used as implementation references. Redistribution and use must continue to comply with the applicable upstream terms.

## ONNX Runtime

Microsoft ONNX Runtime: https://github.com/microsoft/onnxruntime  
License: MIT.

The iOS Swift Package Manager distribution is linked at build time. No model is supplied by ONNX Runtime.

## Runtime-downloaded models

Models are not bundled in this repository or the IPA. The app downloads them only when required and retains the upstream vendor/license metadata in its catalogue.

- HyperSwap 1a 256: FaceFusion, ResearchRAIL.
- ArcFace W600K R50: InsightFace, Non-Commercial as identified by FaceFusion upstream.
- GPEN BFR 512: yangxy, Apache-2.0 as identified by FaceFusion upstream.
- FRAN: ry-lu, MIT.
- FairFace: dchen236, CC-BY-4.0.
- MODNet: ZHKKKe, Apache-2.0.
- DeOldify Stable: jantic, MIT as identified by FaceFusion upstream.
- Real-ESRGAN x2 FP16: xinntao, BSD-3-Clause.
- LivePortrait components: KwaiVGI, MIT.
- DeepFaceLive DFM model `iperov/elon_musk_224`: downloaded from FaceFusion's published Hugging Face model collection; model-specific terms must be reviewed and followed.

Users and redistributors are responsible for complying with model-specific restrictions, including research or non-commercial restrictions where applicable.

## UI design reference

The interface follows native iOS conventions plus accessibility, touch-target, hierarchy, and progressive-disclosure guidance inspected from `nextlevelbuilder/ui-ux-pro-max-skill`. No code or runtime dependency from that repository is bundled.
