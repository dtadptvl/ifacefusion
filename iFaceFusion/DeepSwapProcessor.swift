import UIKit
import CoreImage

actor DeepSwapProcessor {
    private let models = ModelStore.shared
    private let engine = ONNXEngine.shared

    func process(image: UIImage, morphPercent: Double) async throws -> UIImage {
        guard let asset = ModelCatalog.asset(processor: .deepSwap, role: "deep_swapper") else {
            throw PipelineError.modelMissing("Deep Swap")
        }
        let modelURL = try await models.ensure(asset)
        let geometry = try await FaceGeometryDetector.detect(in: image)
        let (crop, warp) = try ImageWarp.crop(
            image,
            geometry: geometry,
            template: .dflWholeFace,
            size: CGSize(width: 224, height: 224)
        )

        let prepared = crop.unsharp(radius: 2, intensity: 0.75)
        let inputNames = try await engine.inputNames(modelURL: modelURL)
        guard let faceName = inputNames.first(where: { $0.lowercased().contains("in_face") || $0.lowercased().contains("face") }) else {
            throw InferenceError.unsupported("The selected DFM model did not expose a face input.")
        }

        var inputs: [String: FloatTensor] = [
            faceName: FloatTensor(
                values: try TensorImage.nhwcBGR(
                    prepared,
                    size: CGSize(width: 224, height: 224)
                ),
                shape: [1, 224, 224, 3]
            )
        ]

        if let morphName = inputNames.first(where: { $0.lowercased().contains("morph") }) {
            inputs[morphName] = FloatTensor(
                values: [Float(max(0, min(100, morphPercent)) / 100)],
                shape: [1]
            )
        }

        let orderedOutputs = try await engine.outputNames(modelURL: modelURL)
        let outputs = try await engine.runFloatModel(
            modelURL: modelURL,
            inputs: inputs,
            requestedOutputs: orderedOutputs
        )
        guard orderedOutputs.count >= 2, let faceTensor = outputs[orderedOutputs[1]] else {
            throw InferenceError.io
        }

        let swapped = try TensorImage.imageNHWCBGR(faceTensor)
        return try ImageWarp.paste(swapped, onto: image, warp: warp)
    }
}

extension UIImage {
    func unsharp(radius: Double, intensity: Double) -> UIImage {
        guard let cgImage = normalizedCGImage else {
            return self
        }
        let input = CIImage(cgImage: cgImage)
        let output = input.applyingFilter(
            "CIUnsharpMask",
            parameters: [
                kCIInputRadiusKey: radius,
                kCIInputIntensityKey: intensity
            ]
        )
        guard let rendered = ImageWarp.context.createCGImage(output, from: input.extent) else {
            return self
        }
        return UIImage(cgImage: rendered)
    }
}

extension TensorImage {
    static func nhwcBGR(_ image: UIImage, size: CGSize) throws -> [Float] {
        guard let cgImage = image.normalizedCGImage else {
            throw InferenceError.io
        }
        let width = Int(size.width)
        let height = Int(size.height)
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        guard let context = CGContext(
            data: &rgba,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw InferenceError.io
        }
        context.interpolationQuality = .high
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        var output = [Float](repeating: 0, count: width * height * 3)
        for index in 0..<(width * height) {
            output[index * 3] = Float(rgba[index * 4 + 2]) / 255
            output[index * 3 + 1] = Float(rgba[index * 4 + 1]) / 255
            output[index * 3 + 2] = Float(rgba[index * 4]) / 255
        }
        return output
    }

    static func imageNHWCBGR(_ tensor: FloatTensor) throws -> UIImage {
        guard tensor.shape.count >= 4 else {
            throw InferenceError.io
        }
        let height = tensor.shape[tensor.shape.count - 3]
        let width = tensor.shape[tensor.shape.count - 2]
        guard tensor.values.count >= width * height * 3 else {
            throw InferenceError.io
        }

        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        for index in 0..<(width * height) {
            let base = index * 3
            rgba[index * 4] = UInt8(max(0, min(255, Int(tensor.values[base + 2] * 255))))
            rgba[index * 4 + 1] = UInt8(max(0, min(255, Int(tensor.values[base + 1] * 255))))
            rgba[index * 4 + 2] = UInt8(max(0, min(255, Int(tensor.values[base] * 255))))
        }

        guard
            let provider = CGDataProvider(data: Data(rgba) as CFData),
            let cgImage = CGImage(
                width: width,
                height: height,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: width * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                provider: provider,
                decode: nil,
                shouldInterpolate: true,
                intent: .defaultIntent
            )
        else {
            throw InferenceError.io
        }
        return UIImage(cgImage: cgImage)
    }
}
