import UIKit
import CoreImage

actor ProcessingPipeline {
    private let models=ModelStore.shared
    private let engine=ONNXEngine.shared

    func process(source:UIImage?,target:UIImage,processors:[ProcessorID],settings:ProcessorSettings,progress:@escaping @Sendable(String)->Void) async throws -> UIImage {
        _ = try await FaceGeometryDetector.detect(in:target)
        if processors.contains(.faceSwap) {
            guard let source else { throw PipelineError.sourceRequired }
            _ = try await FaceGeometryDetector.detect(in:source)
        }
        var image=target
        for processor in processors {
            progress(processor.title)
            switch processor {
            case .faceSwap:
                image=try await faceSwap(source:source!,target:image)
            case .faceEnhance:
                image=try await faceEnhance(image,blend:settings.faceEnhanceBlend)
            case .backgroundRemove:
                image=try await removeBackground(image,opacity:settings.backgroundOpacity)
            case .frameEnhance:
                image=try await enhanceFrame(image,blend:settings.frameEnhanceBlend)
            case .colourise:
                image=try await colourise(image,blend:settings.colourBlend)
            case .age:
                image=try await age(image,direction:settings.ageDirection)
            case .expressionRestore:
                throw PipelineError.specialised("Expression Restore models are catalogued and cached on demand, but the LivePortrait multi-tensor motion pipeline is not yet enabled in this build.")
            case .faceEdit:
                throw PipelineError.specialised("Face Edit models are catalogued and cached on demand, but the LivePortrait multi-tensor edit pipeline is not yet enabled in this build.")
            case .deepSwap:
                throw PipelineError.specialised("Deep Swap uses DFM models rather than ONNX and cannot run in the selected iOS ONNX runtime.")
            }
        }
        return image
    }

    private func asset(_ p:ProcessorID)throws->ModelAsset {
        guard let a=ModelCatalog.required(for:p).first else { throw PipelineError.modelMissing(p.title) }; return a
    }

    private func faceEnhance(_ image:UIImage,blend:Double) async throws -> UIImage {
        let a=try asset(.faceEnhance),url=try await models.ensure(a)
        let g=try await FaceGeometryDetector.detect(in:image)
        let (crop,warp)=try ImageWarp.crop(image,geometry:g,template:.ffhq512,size:.init(width:512,height:512))
        let enhanced=try await engine.runImage(modelURL:url,image:crop,inputSize:.init(width:512,height:512),normalization:-1...1)
        return try ImageWarp.paste(enhanced,onto:image,warp:warp)
    }

    private func faceSwap(source:UIImage,target:UIImage) async throws -> UIImage {
        let a=try asset(.faceSwap); _=try await models.ensure(a)
        // HyperSwap needs ArcFace identity embeddings in addition to its image tensor. Keep this explicit instead of fabricating an identity.
        throw PipelineError.specialised("Face Swap downloaded HyperSwap successfully, but its ArcFace embedding input is not yet wired in this build.")
    }

    private func removeBackground(_ image:UIImage,opacity:Double) async throws -> UIImage {
        let a=try asset(.backgroundRemove),url=try await models.ensure(a)
        let matte=try await engine.runImage(modelURL:url,image:image,inputSize:.init(width:512,height:512),normalization:-1...1)
        guard let base=image.normalizedCGImage,let mask=matte.normalizedCGImage else{return image}
        let ci=CIImage(cgImage:base),m=CIImage(cgImage:mask).applyingFilter("CIColorControls",parameters:[kCIInputSaturationKey:0])
        let clear=CIImage(color:.clear).cropped(to:ci.extent)
        let result=CIFilter(name:"CIBlendWithMask",parameters:[kCIInputImageKey:ci,kCIInputBackgroundImageKey:clear,kCIInputMaskImageKey:m])?.outputImage ?? ci
        guard let cg=ImageWarp.context.createCGImage(result,from:ci.extent) else{return image}; return UIImage(cgImage:cg)
    }

    private func enhanceFrame(_ image:UIImage,blend:Double) async throws -> UIImage {
        let a=try asset(.frameEnhance),url=try await models.ensure(a)
        // Real-ESRGAN is tiled upstream; run a bounded full image here only after resizing long edge to 1024 to control memory.
        let input=image.resized(maxLongEdge:1024)
        return try await engine.runImage(modelURL:url,image:input,inputSize:input.size,normalization:0...1)
    }

    private func colourise(_ image:UIImage,blend:Double) async throws -> UIImage {
        let a=try asset(.colourise),url=try await models.ensure(a)
        return try await engine.runImage(modelURL:url,image:image,inputSize:.init(width:256,height:256),normalization:0...1)
    }

    private func age(_ image:UIImage,direction:Double) async throws -> UIImage {
        let a=try asset(.age); _=try await models.ensure(a)
        throw PipelineError.specialised("FRAN downloaded successfully, but its direction-conditioned output requires specialised multi-input preprocessing not yet enabled in this build.")
    }
}
enum PipelineError:LocalizedError {
    case sourceRequired,modelMissing(String),specialised(String)
    var errorDescription:String? {
        switch self {
        case .sourceRequired:"Choose a Source image before processing."
        case .modelMissing(let p):"No model is configured for \(p)."
        case .specialised(let s):s
        }
    }
}
extension UIImage {
    func resized(maxLongEdge:CGFloat)->UIImage {
        let scale=min(1,maxLongEdge/max(size.width,size.height)); guard scale<1 else{return self}
        let s=CGSize(width:round(size.width*scale),height:round(size.height*scale))
        return UIGraphicsImageRenderer(size:s).image{_ in draw(in:CGRect(origin:.zero,size:s))}
    }
}