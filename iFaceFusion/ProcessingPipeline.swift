import UIKit
import CoreImage

actor ProcessingPipeline {
    private let models = ModelStore.shared
    private let engine = ONNXEngine.shared
    private let livePortrait = LivePortraitProcessor()
    private let deepSwapProcessor = DeepSwapProcessor()

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
                image = try await livePortrait.restoreExpression(
                    originalTarget: target,
                    currentImage: image,
                    factorPercent: settings.expressionFactor
                )
            case .faceEdit:
                image = try await livePortrait.editFace(
                    image: image,
                    settings: settings
                )
            case .deepSwap:
                image = try await deepSwapProcessor.process(
                    image: image,
                    morphPercent: settings.deepSwapMorph
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
        let matte = try await engine.runMask(
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

        let modelTile = 256
        let pad = 16
        let seam = 8
        let scale = 2
        let core = modelTile - 2 * seam

        let sourceSize = image.pixelSize
        let sourceWidth = Int(sourceSize.width)
        let sourceHeight = Int(sourceSize.height)
        let topPad = pad + seam

        func tailPad(_ length: Int) -> Int {
            let remainder = (length + 2 * pad) % core
            return topPad + core - remainder
        }

        let bottomPad = tailPad(sourceHeight)
        let rightPad = tailPad(sourceWidth)
        let paddedWidth = sourceWidth + topPad + rightPad
        let paddedHeight = sourceHeight + topPad + bottomPad

        let padded = image.drawn(
            canvas: CGSize(width: paddedWidth, height: paddedHeight),
            origin: CGPoint(x: topPad, y: topPad)
        )

        var enhancedTiles: [UIImage] = []
        var row = seam
        while row < paddedHeight - seam {
            var column = seam
            while column < paddedWidth - seam {
                let rect = CGRect(
                    x: column - seam,
                    y: row - seam,
                    width: modelTile,
                    height: modelTile
                )
                let tile = try padded.cropped(to: rect)
                let enhanced = try await engine.runImage(
                    modelURL: url,
                    image: tile,
                    inputSize: CGSize(width: modelTile, height: modelTile),
                    normalization: 0...1
                )
                enhancedTiles.append(enhanced)
                column += core
            }
            row += core
        }

        let mergedWidth = paddedWidth * scale
        let mergedHeight = paddedHeight * scale
        let scaledSeam = seam * scale
        let tileCore = core * scale
        let tilesPerRow = paddedWidth / core
        let rendererFormat = UIGraphicsImageRendererFormat()
        rendererFormat.scale = 1
        rendererFormat.opaque = false
        let renderer = UIGraphicsImageRenderer(
            size: CGSize(width: mergedWidth, height: mergedHeight),
            format: rendererFormat
        )
        let merged = renderer.image { _ in
            for (index, tile) in enhancedTiles.enumerated() {
                guard let center = try? tile.cropped(to: CGRect(
                    x: scaledSeam,
                    y: scaledSeam,
                    width: tileCore,
                    height: tileCore
                )) else {
                    continue
                }
                let rowIndex = index / tilesPerRow
                let columnIndex = index % tilesPerRow
                center.draw(in: CGRect(
                    x: columnIndex * tileCore,
                    y: rowIndex * tileCore,
                    width: tileCore,
                    height: tileCore
                ))
            }
        }

        let output = try merged.cropped(to: CGRect(
            x: pad * scale,
            y: pad * scale,
            width: sourceWidth * scale,
            height: sourceHeight * scale
        ))
        let originalUpscaled = image.resized(to: output.pixelSize)
        return ImageBlend.mix(original: originalUpscaled, processed: output, amount: blend)
    }

    private func colourise(_ image: UIImage, blend: Double) async throws -> UIImage {
        let asset = try asset(.colourise, role: "frame_colorizer")
        let url = try await models.ensure(asset)
        let grayscale = image.monochrome()
        let coloured = try await engine.runImage(
            modelURL: url,
            image: grayscale,
            inputSize: CGSize(width: 256, height: 256),
            normalization: 0...255,
            outputRange: 0...255
        )
        let restoredSize = coloured.resized(to: image.pixelSize)

        guard
            let originalCG = image.normalizedCGImage,
            let colourCG = restoredSize.normalizedCGImage
        else {
            return restoredSize
        }
        let originalCI = CIImage(cgImage: originalCG)
        let colourCI = CIImage(cgImage: colourCG)
        let luminancePreserved = CIFilter(
            name: "CIColorBlendMode",
            parameters: [
                kCIInputImageKey: colourCI,
                kCIInputBackgroundImageKey: originalCI
            ]
        )?.outputImage ?? colourCI
        guard let outputCG = ImageWarp.context.createCGImage(
            luminancePreserved,
            from: originalCI.extent
        ) else {
            return restoredSize
        }
        let output = UIImage(cgImage: outputCG)
        return ImageBlend.mix(original: image, processed: output, amount: blend)
    }

    private func age(_ image: UIImage, direction: Double) async throws -> UIImage {
        let modifierAsset = try asset(.age, role: "age_modifier")
        let classifierAsset = try asset(.age, role: "face_classifier")
        let modifierURL = try await models.ensure(modifierAsset)
        let classifierURL = try await models.ensure(classifierAsset)

        let geometry = try await FaceGeometryDetector.detect(in: image)
        let (classifierCrop, _) = try ImageWarp.crop(
            image,
            geometry: geometry,
            template: .arcface112v2,
            size: CGSize(width: 224, height: 224)
        )
        let ageClass = try await engine.classifyAge(
            modelURL: classifierURL,
            image: classifierCrop,
            inputSize: CGSize(width: 224, height: 224)
        )
        let currentAge = ageMidpoint(for: ageClass)
        let destinationAge = max(0, min(100, currentAge + direction))
        let directionTensor: [Float] = [
            Float(currentAge / 100),
            Float(destinationAge / 100)
        ]

        let (crop, warp) = try ImageWarp.crop(
            image,
            geometry: geometry,
            template: .ffhq512,
            size: CGSize(width: 1024, height: 1024)
        )
        let modified = try await engine.modifyAge(
            modelURL: modifierURL,
            image: crop,
            inputSize: CGSize(width: 1024, height: 1024),
            direction: directionTensor
        )
        return try ImageWarp.paste(modified, onto: image, warp: warp)
    }

    private func ageMidpoint(for ageClass: Int) -> Double {
        switch ageClass {
        case 0: return 0.5
        case 1: return 5.5
        case 2: return 14.5
        case 3: return 24.5
        case 4: return 34.5
        case 5: return 44.5
        case 6: return 54.5
        case 7: return 64.5
        default: return 84.5
        }
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
    func cropped(to rect: CGRect) throws -> UIImage {
        guard let cgImage = normalizedCGImage?.cropping(to: rect.integral) else {
            throw InferenceError.io
        }
        return UIImage(cgImage: cgImage)
    }

    func drawn(canvas: CGSize, origin: CGPoint) -> UIImage {
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = false
        return UIGraphicsImageRenderer(size: canvas, format: format).image { _ in
            draw(in: CGRect(origin: origin, size: pixelSize))
        }
    }

    var pixelSize: CGSize {
        guard let cgImage = normalizedCGImage else {
            return size
        }
        return CGSize(width: cgImage.width, height: cgImage.height)
    }

    func monochrome() -> UIImage {
        guard let cgImage = normalizedCGImage else {
            return self
        }
        let input = CIImage(cgImage: cgImage)
        let output = input.applyingFilter(
            "CIColorControls",
            parameters: [kCIInputSaturationKey: 0]
        )
        guard let rendered = ImageWarp.context.createCGImage(output, from: input.extent) else {
            return self
        }
        return UIImage(cgImage: rendered)
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
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = false
        return UIGraphicsImageRenderer(size: targetSize, format: format).image { _ in
            draw(in: CGRect(origin: .zero, size: targetSize))
        }
    }
}