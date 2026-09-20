import UIKit
import CoreImage

actor ProcessingPipeline {
    private let models = ModelStore.shared
    private let engine = ONNXEngine.shared

    func process(
        source: UIImage?,
        target: UIImage,
        processors: [ProcessorID],
        settings: ProcessorSettings,
        progress: @escaping @Sendable (String) -> Void
    ) async throws -> UIImage {
        _ = try await FaceGeometryDetector.detect(in: target)

        if processors.contains(.faceSwap) {
            guard let source else {
                throw PipelineError.sourceRequired
            }
            _ = try await FaceGeometryDetector.detect(in: source)
        }

        var image = target
        for processor in processors {
            progress(processor.title)
            switch processor {
            case .faceSwap:
                guard let source else {
                    throw PipelineError.sourceRequired
                }
                image = try await faceSwap(
                    source: source,
                    target: image,
                    weight: settings.faceSwapWeight
                )
            case .faceEnhance:
                image = try await faceEnhance(image, blend: settings.faceEnhanceBlend)
            case .backgroundRemove:
                image = try await removeBackground(image, opacity: settings.backgroundOpacity)
            case .frameEnhance:
                image = try await enhanceFrame(image, blend: settings.frameEnhanceBlend)
            case .colourise:
                image = try await colourise(image, blend: settings.colourBlend)
            case .age:
                image = try await age(image, direction: settings.ageDirection)
            case .expressionRestore:
                throw PipelineError.specialised(
                    "Expression Restore models are available, but the LivePortrait multi-tensor motion pipeline is not yet enabled in this build."
                )
            case .faceEdit:
                throw PipelineError.specialised(
                    "Face Edit models are available, but the LivePortrait multi-tensor edit pipeline is not yet enabled in this build."
                )
            case .deepSwap:
                throw PipelineError.specialised(
                    "Deep Swap uses DFM models rather than FaceFusion's ONNX processor path and is not compatible with this runtime."
                )
            }
        }
        return image
    }

    private func asset(_ processor: ProcessorID, role: String) throws -> ModelAsset {
        guard let asset = ModelCatalog.asset(processor: processor, role: role) else {
            throw PipelineError.modelMissing("\(processor.title) / \(role)")
        }
        return asset
    }

    private func faceSwap(
        source: UIImage,
        target: UIImage,
        weight: Double
    ) async throws -> UIImage {
        let swapAsset = try asset(.faceSwap, role: "face_swapper")
        let recognizerAsset = try asset(.faceSwap, role: "face_recognizer")
        let swapURL = try await models.ensure(swapAsset)
        let recognizerURL = try await models.ensure(recognizerAsset)

        let sourceGeometry = try await FaceGeometryDetector.detect(in: source)
        let targetGeometry = try await FaceGeometryDetector.detect(in: target)

        let (sourceArcFace, _) = try ImageWarp.crop(
            source,
            geometry: sourceGeometry,
            template: .arcface112v2,
            size: CGSize(width: 112, height: 112)
        )
        let (targetArcFace, _) = try ImageWarp.crop(
            target,
            geometry: targetGeometry,
            template: .arcface112v2,
            size: CGSize(width: 112, height: 112)
        )

        let rawSource = try await engine.embedding(
            modelURL: recognizerURL,
            image: sourceArcFace,
            inputSize: CGSize(width: 112, height: 112)
        )
        let rawTarget = try await engine.embedding(
            modelURL: recognizerURL,
            image: targetArcFace,
            inputSize: CGSize(width: 112, height: 112)
        )
        let sourceEmbedding = normalized(rawSource)
        let targetEmbedding = normalized(rawTarget)

        let upstreamWeight = Float(0.35 - 0.70 * max(0, min(1, weight)))
        let balanced = zip(sourceEmbedding, targetEmbedding).map {
            $0 * (1 - upstreamWeight) + $1 * upstreamWeight
        }

        let (targetCrop, warp) = try ImageWarp.crop(
            target,
            geometry: targetGeometry,
            template: .arcface128,
            size: CGSize(width: 256, height: 256)
        )
        let swapped = try await engine.faceSwap(
            modelURL: swapURL,
            sourceEmbedding: balanced,
            targetImage: targetCrop,
            inputSize: CGSize(width: 256, height: 256)
        )
        return try ImageWarp.paste(swapped, onto: target, warp: warp)
    }

    private func faceEnhance(_ image: UIImage, blend: Double) async throws -> UIImage {
        let asset = try asset(.faceEnhance, role: "face_enhancer")
        let url = try await models.ensure(asset)
        let geometry = try await FaceGeometryDetector.detect(in: image)
        let (crop, warp) = try ImageWarp.crop(
            image,
            geometry: geometry,
            template: .ffhq512,
            size: CGSize(width: 512, height: 512)
        )
        let enhanced = try await engine.runImage(
            modelURL: url,
            image: crop,
            inputSize: CGSize(width: 512, height: 512),
            normalization: -1...1,
            outputRange: -1...1
        )
        let pasted = try ImageWarp.paste(enhanced, onto: image, warp: warp)
        return ImageBlend.mix(original: image, processed: pasted, amount: blend)
    }

    private func removeBackground(_ image: UIImage, opacity: Double) async throws -> UIImage {
        let asset = try asset(.backgroundRemove, role: "background_remover")
        let url = try await models.ensure(asset)
        let matte = try await engine.runImage(
            modelURL: url,
            image: image,
            inputSize: CGSize(width: 512, height: 512),
            normalization: -1...1
        )

        guard let baseCG = image.normalizedCGImage, let maskCG = matte.normalizedCGImage else {
            return image
        }
        let base = CIImage(cgImage: baseCG)
        let mask = CIImage(cgImage: maskCG)
            .applyingFilter("CIColorControls", parameters: [kCIInputSaturationKey: 0])
            .transformed(by: CGAffineTransform(
                scaleX: base.extent.width / CGFloat(maskCG.width),
                y: base.extent.height / CGFloat(maskCG.height)
            ))
            .cropped(to: base.extent)

        let background = CIImage(
            color: CIColor(red: 0, green: 0, blue: 0, alpha: CGFloat(opacity))
        ).cropped(to: base.extent)

        let output = CIFilter(
            name: "CIBlendWithMask",
            parameters: [
                kCIInputImageKey: base,
                kCIInputBackgroundImageKey: background,
                kCIInputMaskImageKey: mask
            ]
        )?.outputImage ?? base

        guard let cgImage = ImageWarp.context.createCGImage(output, from: base.extent) else {
            return image
        }
        return UIImage(cgImage: cgImage)
    }

    private func enhanceFrame(_ image: UIImage, blend: Double) async throws -> UIImage {
        let asset = try asset(.frameEnhance, role: "frame_enhancer")
        let url = try await models.ensure(asset)
        let input = image.resized(maxLongEdge: 1024)
        let enhanced = try await engine.runImage(
            modelURL: url,
            image: input,
            inputSize: input.pixelSize,
            normalization: 0...1
        )
        return enhanced
    }

    private func colourise(_ image: UIImage, blend: Double) async throws -> UIImage {
        let asset = try asset(.colourise, role: "frame_colorizer")
        let url = try await models.ensure(asset)
        let coloured = try await engine.runImage(
            modelURL: url,
            image: image,
            inputSize: CGSize(width: 256, height: 256),
            normalization: 0...1
        )
        let restoredSize = coloured.resized(to: image.pixelSize)
        return ImageBlend.mix(original: image, processed: restoredSize, amount: blend)
    }

    private func age(_ image: UIImage, direction: Double) async throws -> UIImage {
        let asset = try asset(.age, role: "age_modifier")
        _ = try await models.ensure(asset)
        throw PipelineError.specialised(
            "FRAN is cached successfully, but its direction-conditioned target/background inputs are not yet enabled in this build."
        )
    }

    private func normalized(_ values: [Float]) -> [Float] {
        let norm = sqrt(values.reduce(Float.zero) { $0 + $1 * $1 })
        guard norm > 0 else {
            return values
        }
        return values.map { $0 / norm }
    }
}

enum PipelineError: LocalizedError {
    case sourceRequired
    case modelMissing(String)
    case specialised(String)

    var errorDescription: String? {
        switch self {
        case .sourceRequired:
            return "Choose a Source image before processing."
        case .modelMissing(let processor):
            return "No model is configured for \(processor)."
        case .specialised(let message):
            return message
        }
    }
}

enum ImageBlend {
    static func mix(original: UIImage, processed: UIImage, amount: Double) -> UIImage {
        let amount = max(0, min(1, amount))
        guard
            let originalCG = original.normalizedCGImage,
            let processedCG = processed.normalizedCGImage
        else {
            return processed
        }

        let originalCI = CIImage(cgImage: originalCG)
        var processedCI = CIImage(cgImage: processedCG)
        if processedCI.extent.size != originalCI.extent.size {
            processedCI = processedCI.transformed(by: CGAffineTransform(
                scaleX: originalCI.extent.width / processedCI.extent.width,
                y: originalCI.extent.height / processedCI.extent.height
            ))
        }

        let alpha = processedCI.applyingFilter(
            "CIColorMatrix",
            parameters: [
                "inputAVector": CIVector(x: 0, y: 0, z: 0, w: amount)
            ]
        )
        let output = alpha.composited(over: originalCI)
        guard let cgImage = ImageWarp.context.createCGImage(output, from: originalCI.extent) else {
            return processed
        }
        return UIImage(cgImage: cgImage)
    }
}

extension UIImage {
    var pixelSize: CGSize {
        guard let cgImage = normalizedCGImage else {
            return size
        }
        return CGSize(width: cgImage.width, height: cgImage.height)
    }

    func resized(maxLongEdge: CGFloat) -> UIImage {
        let pixels = pixelSize
        let scale = min(1, maxLongEdge / max(pixels.width, pixels.height))
        guard scale < 1 else {
            return self
        }
        return resized(to: CGSize(
            width: round(pixels.width * scale),
            height: round(pixels.height * scale)
        ))
    }

    func resized(to targetSize: CGSize) -> UIImage {
        UIGraphicsImageRenderer(size: targetSize).image { _ in
            draw(in: CGRect(origin: .zero, size: targetSize))
        }
    }
}