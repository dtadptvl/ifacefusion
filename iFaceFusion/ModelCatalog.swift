import Foundation

struct ModelAsset: Identifiable, Hashable {
    let id, file, release: String
    let processor: ProcessorID
    let vendor, license: String
    let approximateMB: Int
    let role: String
    var remoteURL: URL { URL(string: "https://github.com/facefusion/facefusion-assets/releases/download/\(release)/\(file)")! }
}
enum ModelCatalog {
    static let assets: [ModelAsset] = [
        .init(id:"hyperswap_1a_256",file:"hyperswap_1a_256.onnx",release:"models-3.3.0",processor:.faceSwap,vendor:"FaceFusion",license:"ResearchRAIL",approximateMB:384,role:"face_swapper"),
        .init(id:"gpen_bfr_512",file:"gpen_bfr_512.onnx",release:"models-3.0.0",processor:.faceEnhance,vendor:"yangxy",license:"Apache-2.0",approximateMB:271,role:"face_enhancer"),
        .init(id:"fran",file:"fran.onnx",release:"models-3.6.0",processor:.age,vendor:"ry-lu",license:"MIT",approximateMB:125,role:"age_modifier"),
        .init(id:"modnet",file:"modnet.onnx",release:"models-3.5.0",processor:.backgroundRemove,vendor:"ZHKKKe",license:"Apache-2.0",approximateMB:25,role:"background_remover"),
        .init(id:"ddcolor",file:"ddcolor.onnx",release:"models-3.0.0",processor:.colourise,vendor:"piddnad",license:"Apache-2.0",approximateMB:935,role:"frame_colorizer"),
        .init(id:"real_esrgan_x2_fp16",file:"real_esrgan_x2_fp16.onnx",release:"models-3.0.0",processor:.frameEnhance,vendor:"xinntao",license:"BSD-3-Clause",approximateMB:35,role:"frame_enhancer"),
        .init(id:"live_portrait_feature_extractor",file:"live_portrait_feature_extractor.onnx",release:"models-3.0.0",processor:.faceEdit,vendor:"KwaiVGI",license:"MIT",approximateMB:4,role:"feature_extractor"),
        .init(id:"live_portrait_motion_extractor",file:"live_portrait_motion_extractor.onnx",release:"models-3.0.0",processor:.faceEdit,vendor:"KwaiVGI",license:"MIT",approximateMB:108,role:"motion_extractor"),
        .init(id:"live_portrait_eye_retargeter",file:"live_portrait_eye_retargeter.onnx",release:"models-3.0.0",processor:.faceEdit,vendor:"KwaiVGI",license:"MIT",approximateMB:1,role:"eye_retargeter"),
        .init(id:"live_portrait_lip_retargeter",file:"live_portrait_lip_retargeter.onnx",release:"models-3.0.0",processor:.faceEdit,vendor:"KwaiVGI",license:"MIT",approximateMB:1,role:"lip_retargeter"),
        .init(id:"live_portrait_stitcher",file:"live_portrait_stitcher.onnx",release:"models-3.0.0",processor:.faceEdit,vendor:"KwaiVGI",license:"MIT",approximateMB:1,role:"stitcher"),
        .init(id:"live_portrait_generator",file:"live_portrait_generator.onnx",release:"models-3.0.0",processor:.faceEdit,vendor:"KwaiVGI",license:"MIT",approximateMB:213,role:"generator")
    ]
    static func required(for p: ProcessorID) -> [ModelAsset] {
        if p == .expressionRestore { return assets.filter { $0.processor == .faceEdit && ["feature_extractor","motion_extractor","generator"].contains($0.role) } }
        if p == .deepSwap { return [] }
        return assets.filter { $0.processor == p }
    }
}