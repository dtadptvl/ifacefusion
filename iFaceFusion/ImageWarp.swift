import UIKit
import CoreImage

enum WarpTemplate {
    case arcface112v2
    case arcface128
    case dflWholeFace
    case ffhq512

    var normalized: [CGPoint] {
        switch self {
        case .arcface112v2:
            return [
                .init(x: 0.34191607, y: 0.46157411),
                .init(x: 0.65653393, y: 0.45983393),
                .init(x: 0.50022500, y: 0.64050536),
                .init(x: 0.37097589, y: 0.82469196),
                .init(x: 0.63151696, y: 0.82325089)
            ]
        case .arcface128:
            return [
                .init(x: 0.36167656, y: 0.40387734),
                .init(x: 0.63696719, y: 0.40235469),
                .init(x: 0.50019687, y: 0.56044219),
                .init(x: 0.38710391, y: 0.72160547),
                .init(x: 0.61507734, y: 0.72034453)
            ]
        case .dflWholeFace:
            return [
                .init(x: 0.35342266, y: 0.39285716),
                .init(x: 0.62797622, y: 0.39285716),
                .init(x: 0.48660713, y: 0.54017860),
                .init(x: 0.38839287, y: 0.68750011),
                .init(x: 0.59821427, y: 0.68750011)
            ]
        case .ffhq512:
            return [
                .init(x: 0.37691676, y: 0.46864664),
                .init(x: 0.62285697, y: 0.46912813),
                .init(x: 0.50123859, y: 0.61331904),
                .init(x: 0.39308822, y: 0.72541100),
                .init(x: 0.61150205, y: 0.72490465)
            ]
        }
    }
}

struct AffineWarp {
    let transform: CGAffineTransform
    let inverse: CGAffineTransform
}

enum ImageWarp {
    static let context = CIContext(options: [.cacheIntermediates: false])

    static func estimate(source: [CGPoint], template: WarpTemplate, size: CGSize) -> AffineWarp {
        let target = template.normalized.map {
            CGPoint(x: $0.x * size.width, y: $0.y * size.height)
        }
        let s0 = source[0]
        let s1 = source[1]
        let t0 = target[0]
        let t1 = target[1]
        let sv = CGPoint(x: s1.x - s0.x, y: s1.y - s0.y)
        let tv = CGPoint(x: t1.x - t0.x, y: t1.y - t0.y)
        let scale = hypot(tv.x, tv.y) / max(hypot(sv.x, sv.y), 0.001)
        let angle = atan2(tv.y, tv.x) - atan2(sv.y, sv.x)

        var transform = CGAffineTransform(scaleX: scale, y: scale).rotated(by: angle)
        let mapped = source.map { $0.applying(transform) }
        let dx = zip(mapped, target).map { $1.x - $0.x }.reduce(0, +) / CGFloat(source.count)
        let dy = zip(mapped, target).map { $1.y - $0.y }.reduce(0, +) / CGFloat(source.count)
        transform = transform.concatenating(CGAffineTransform(translationX: dx, y: dy))
        return AffineWarp(transform: transform, inverse: transform.inverted())
    }

    static func crop(
        _ image: UIImage,
        geometry: FaceGeometry,
        template: WarpTemplate,
        size: CGSize
    ) throws -> (UIImage, AffineWarp) {
        guard let cgImage = image.normalizedCGImage else {
            throw FaceGeometryError.noFace
        }
        let warp = estimate(source: geometry.landmarks, template: template, size: size)
        let ciImage = CIImage(cgImage: cgImage).transformed(by: warp.transform)
        let rect = CGRect(origin: .zero, size: size)
        guard let output = context.createCGImage(ciImage, from: rect) else {
            throw FaceGeometryError.noFace
        }
        return (UIImage(cgImage: output), warp)
    }

    static func paste(_ crop: UIImage, onto base: UIImage, warp: AffineWarp) throws -> UIImage {
        guard let baseCG = base.normalizedCGImage, let cropCG = crop.normalizedCGImage else {
            return base
        }
        let baseCI = CIImage(cgImage: baseCG)
        let cropCI = CIImage(cgImage: cropCG).transformed(by: warp.inverse)
        let extent = cropCI.extent
        let radius = min(extent.width, extent.height) * 0.08
        guard let mask = CIFilter(
            name: "CIRoundedRectangleGenerator",
            parameters: [
                "inputExtent": CIVector(cgRect: extent.insetBy(dx: radius * 0.2, dy: radius * 0.2)),
                "inputRadius": radius
            ]
        )?.outputImage?.cropped(to: extent) else {
            return base
        }

        let blended = CIFilter(
            name: "CIBlendWithMask",
            parameters: [
                kCIInputImageKey: cropCI,
                kCIInputBackgroundImageKey: baseCI,
                kCIInputMaskImageKey: mask
            ]
        )?.outputImage ?? baseCI

        guard let output = context.createCGImage(blended, from: baseCI.extent) else {
            return base
        }
        return UIImage(cgImage: output)
    }
}