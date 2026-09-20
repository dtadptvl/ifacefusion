import UIKit

actor LivePortraitProcessor {
    private let models = ModelStore.shared
    private let engine = ONNXEngine.shared

    struct MotionState {
        var pitch: Float
        var yaw: Float
        var roll: Float
        var scale: Float
        var translation: [Float]
        var expression: [Float]
        var motionPoints: [Float]
    }

    func restoreExpression(
        originalTarget: UIImage,
        currentImage: UIImage,
        factorPercent: Double
    ) async throws -> UIImage {
        let featureURL = try await modelURL(role: "feature_extractor")
        let motionURL = try await modelURL(role: "motion_extractor")
        let generatorURL = try await modelURL(role: "generator")

        let geometry = try await FaceGeometryDetector.detect(in: originalTarget)
        let (targetCrop512, _) = try ImageWarp.crop(
            originalTarget,
            geometry: geometry,
            template: .arcface128,
            size: CGSize(width: 512, height: 512)
        )
        let (currentCrop512, warp) = try ImageWarp.crop(
            currentImage,
            geometry: geometry,
            template: .arcface128,
            size: CGSize(width: 512, height: 512)
        )

        let targetCrop = targetCrop512.resized(to: CGSize(width: 256, height: 256))
        let currentCrop = currentCrop512.resized(to: CGSize(width: 256, height: 256))

        let feature = try await extractFeature(modelURL: featureURL, image: currentCrop)
        let targetMotion = try await extractMotion(modelURL: motionURL, image: targetCrop)
        let currentMotion = try await extractMotion(modelURL: motionURL, image: currentCrop)

        var targetExpression = targetMotion.expression
        let fixedIndices = [0, 4, 5, 8, 9]
        for point in fixedIndices {
            for axis in 0..<3 {
                targetExpression[point * 3 + axis] = currentMotion.expression[point * 3 + axis]
            }
        }

        let factor = Float(max(0, min(100, factorPercent)) / 100 * 1.2)
        for index in targetExpression.indices {
            targetExpression[index] =
                targetExpression[index] * factor +
                currentMotion.expression[index] * (1 - factor)
        }

        let rotation = eulerRotation(
            pitch: currentMotion.pitch,
            yaw: currentMotion.yaw,
            roll: currentMotion.roll
        )
        let targetPoints = transformMotion(
            currentMotion.motionPoints,
            expression: targetExpression,
            rotation: rotation,
            scale: currentMotion.scale,
            translation: currentMotion.translation
        )
        let currentPoints = transformMotion(
            currentMotion.motionPoints,
            expression: currentMotion.expression,
            rotation: rotation,
            scale: currentMotion.scale,
            translation: currentMotion.translation
        )

        let generated = try await generate(
            modelURL: generatorURL,
            feature: feature,
            source: targetPoints,
            target: currentPoints
        )
        let generated512 = generated.resized(to: CGSize(width: 512, height: 512))
        return try ImageWarp.paste(generated512, onto: currentImage, warp: warp)
    }

    func editFace(
        image: UIImage,
        settings: ProcessorSettings
    ) async throws -> UIImage {
        let featureURL = try await modelURL(role: "feature_extractor")
        let motionURL = try await modelURL(role: "motion_extractor")
        let stitchURL = try await modelURL(role: "stitcher")
        let generatorURL = try await modelURL(role: "generator")

        let geometry = try await FaceGeometryDetector.detect(in: image)
        let (crop512, warp) = try ImageWarp.crop(
            image,
            geometry: geometry,
            template: .ffhq512,
            size: CGSize(width: 512, height: 512)
        )
        let crop = crop512.resized(to: CGSize(width: 256, height: 256))
        let feature = try await extractFeature(modelURL: featureURL, image: crop)
        var motion = try await extractMotion(modelURL: motionURL, image: crop)

        let targetRotation = eulerRotation(
            pitch: motion.pitch,
            yaw: motion.yaw,
            roll: motion.roll
        )
        let targetPoints = transformMotion(
            motion.motionPoints,
            expression: motion.expression,
            rotation: targetRotation,
            scale: motion.scale,
            translation: motion.translation
        )

        applyGaze(
            &motion.expression,
            horizontal: Float(clampUnit(settings.faceEditGazeHorizontal)),
            vertical: Float(clampUnit(settings.faceEditGazeVertical))
        )
        applyGrim(&motion.expression, amount: Float(clampUnit(settings.faceEditMouthGrim)))
        applyMouthPosition(
            &motion.expression,
            horizontal: Float(clampUnit(settings.faceEditMouthHorizontal)),
            vertical: Float(clampUnit(settings.faceEditMouthVertical))
        )
        applyPout(&motion.expression, amount: Float(clampUnit(settings.faceEditMouthPout)))
        applyPurse(&motion.expression, amount: Float(clampUnit(settings.faceEditMouthPurse)))
        applySmile(&motion.expression, amount: Float(clampUnit(settings.faceEditSmile)))
        applyEyebrow(&motion.expression, amount: Float(clampUnit(settings.faceEditEyebrow)))

        let editedRotation = eulerRotation(
            pitch: limitedAngle(
                base: motion.pitch,
                proposed: motion.pitch + Float(-20 * clampUnit(settings.faceEditHeadPitch)),
                normalMin: -30,
                normalMax: 30
            ),
            yaw: limitedAngle(
                base: motion.yaw,
                proposed: motion.yaw + Float(-60 * clampUnit(settings.faceEditHeadYaw)),
                normalMin: -60,
                normalMax: 60
            ),
            roll: limitedAngle(
                base: motion.roll,
                proposed: motion.roll + Float(15 * clampUnit(settings.faceEditHeadRoll)),
                normalMin: -20,
                normalMax: 20
            )
        )

        var editedPoints = transformMotion(
            motion.motionPoints,
            expression: motion.expression,
            rotation: editedRotation,
            scale: motion.scale,
            translation: motion.translation
        )

        let eyeAmount = Float(clampUnit(settings.faceEditEyeOpen))
        if eyeAmount != 0 {
            let eyeURL = try await modelURL(role: "eye_retargeter")
            let eyeDelta = try await retargetEye(
                modelURL: eyeURL,
                targetPoints: targetPoints,
                leftRatio: Float(geometry.leftEyeOpenRatio),
                rightRatio: Float(geometry.rightEyeOpenRatio),
                amount: eyeAmount
            )
            add(&editedPoints, eyeDelta)
        }

        let lipAmount = Float(clampUnit(settings.faceEditLipOpen))
        if lipAmount != 0 {
            let lipURL = try await modelURL(role: "lip_retargeter")
            let lipDelta = try await retargetLip(
                modelURL: lipURL,
                targetPoints: targetPoints,
                lipRatio: Float(geometry.lipOpenRatio),
                amount: lipAmount
            )
            add(&editedPoints, lipDelta)
        }

        let stitched = try await stitch(
            modelURL: stitchURL,
            source: editedPoints,
            target: targetPoints
        )
        let generated = try await generate(
            modelURL: generatorURL,
            feature: feature,
            source: stitched,
            target: targetPoints
        )
        let generated512 = generated.resized(to: CGSize(width: 512, height: 512))
        return try ImageWarp.paste(generated512, onto: image, warp: warp)
    }

    private func modelURL(role: String) async throws -> URL {
        guard let asset = ModelCatalog.asset(processor: .faceEdit, role: role) else {
            throw PipelineError.modelMissing("LivePortrait / \(role)")
        }
        return try await models.ensure(asset)
    }

    private func imageTensor(_ image: UIImage) throws -> FloatTensor {
        FloatTensor(
            values: try TensorImage.chw(
                image,
                size: CGSize(width: 256, height: 256),
                range: 0...1
            ),
            shape: [1, 3, 256, 256]
        )
    }

    private func extractFeature(modelURL: URL, image: UIImage) async throws -> FloatTensor {
        guard let inputName = try await engine.inputNames(modelURL: modelURL).first else {
            throw InferenceError.io
        }
        let outputs = try await engine.runFloatModel(
            modelURL: modelURL,
            inputs: [inputName: try imageTensor(image)]
        )
        guard let feature = outputs.values.first else {
            throw InferenceError.io
        }
        return feature
    }

    private func extractMotion(modelURL: URL, image: UIImage) async throws -> MotionState {
        guard let inputName = try await engine.inputNames(modelURL: modelURL).first else {
            throw InferenceError.io
        }
        let orderedNames = try await engine.outputNames(modelURL: modelURL)
        let outputs = try await engine.runFloatModel(
            modelURL: modelURL,
            inputs: [inputName: try imageTensor(image)],
            requestedOutputs: orderedNames
        )
        guard orderedNames.count >= 7 else {
            throw InferenceError.unsupported("LivePortrait motion extractor exposed fewer than seven outputs.")
        }

        func tensor(_ keyword: String, fallback: Int) throws -> FloatTensor {
            if let name = orderedNames.first(where: { $0.lowercased().contains(keyword) }),
               let value = outputs[name] {
                return value
            }
            guard let value = outputs[orderedNames[fallback]] else {
                throw InferenceError.io
            }
            return value
        }

        let pitch = try tensor("pitch", fallback: 0)
        let yaw = try tensor("yaw", fallback: 1)
        let roll = try tensor("roll", fallback: 2)
        let scale = try tensor("scale", fallback: 3)
        let translation = try tensor("translation", fallback: 4)
        let expression = try tensor("expression", fallback: 5)
        let motionPoints = try tensor("motion", fallback: 6)

        guard
            let pitchValue = pitch.values.first,
            let yawValue = yaw.values.first,
            let rollValue = roll.values.first,
            let scaleValue = scale.values.first,
            translation.values.count >= 3,
            expression.values.count >= 63,
            motionPoints.values.count >= 63
        else {
            throw InferenceError.io
        }

        return MotionState(
            pitch: pitchValue,
            yaw: yawValue,
            roll: rollValue,
            scale: scaleValue,
            translation: Array(translation.values.prefix(3)),
            expression: Array(expression.values.prefix(63)),
            motionPoints: Array(motionPoints.values.prefix(63))
        )
    }

    private func stitch(
        modelURL: URL,
        source: [Float],
        target: [Float]
    ) async throws -> [Float] {
        let names = try await engine.inputNames(modelURL: modelURL)
        guard
            let sourceName = names.first(where: { $0.lowercased().contains("source") }),
            let targetName = names.first(where: { $0.lowercased().contains("target") })
        else {
            throw InferenceError.io
        }
        let outputs = try await engine.runFloatModel(
            modelURL: modelURL,
            inputs: [
                sourceName: FloatTensor(values: source, shape: [1, 21, 3]),
                targetName: FloatTensor(values: target, shape: [1, 21, 3])
            ]
        )
        guard let tensor = outputs.values.first else {
            throw InferenceError.io
        }
        return Array(tensor.values.prefix(63))
    }

    private func generate(
        modelURL: URL,
        feature: FloatTensor,
        source: [Float],
        target: [Float]
    ) async throws -> UIImage {
        let names = try await engine.inputNames(modelURL: modelURL)
        guard
            let featureName = names.first(where: { $0.lowercased().contains("feature") }),
            let sourceName = names.first(where: { $0.lowercased().contains("source") }),
            let targetName = names.first(where: { $0.lowercased().contains("target") })
        else {
            throw InferenceError.io
        }

        let outputs = try await engine.runFloatModel(
            modelURL: modelURL,
            inputs: [
                featureName: feature,
                sourceName: FloatTensor(values: source, shape: [1, 21, 3]),
                targetName: FloatTensor(values: target, shape: [1, 21, 3])
            ]
        )
        guard let tensor = outputs.values.first else {
            throw InferenceError.io
        }
        return try image(from: tensor)
    }

    private func image(from tensor: FloatTensor) throws -> UIImage {
        let data = tensor.values.withUnsafeBufferPointer {
            NSMutableData(
                bytes: $0.baseAddress,
                length: $0.count * MemoryLayout<Float>.size
            )
        }
        return try TensorImage.image(
            data: data,
            shape: tensor.shape.map { NSNumber(value: $0) },
            range: 0...1
        )
    }

    private func transformMotion(
        _ points: [Float],
        expression: [Float],
        rotation: [Float],
        scale: Float,
        translation: [Float]
    ) -> [Float] {
        var output = [Float](repeating: 0, count: 63)
        for point in 0..<21 {
            let x = points[point * 3]
            let y = points[point * 3 + 1]
            let z = points[point * 3 + 2]
            let rx = rotation[0] * x + rotation[1] * y + rotation[2] * z
            let ry = rotation[3] * x + rotation[4] * y + rotation[5] * z
            let rz = rotation[6] * x + rotation[7] * y + rotation[8] * z
            output[point * 3] = (rx + expression[point * 3]) * scale + translation[0]
            output[point * 3 + 1] = (ry + expression[point * 3 + 1]) * scale + translation[1]
            output[point * 3 + 2] = (rz + expression[point * 3 + 2]) * scale + translation[2]
        }
        return output
    }

    private func eulerRotation(pitch: Float, yaw: Float, roll: Float) -> [Float] {
        let p = pitch * .pi / 180
        let y = yaw * .pi / 180
        let r = roll * .pi / 180
        let cp = cos(p), sp = sin(p)
        let cy = cos(y), sy = sin(y)
        let cr = cos(r), sr = sin(r)

        return [
            cr * cy,
            cr * sy * sp - sr * cp,
            cr * sy * cp + sr * sp,
            sr * cy,
            sr * sy * sp + cr * cp,
            sr * sy * cp - cr * sp,
            -sy,
            cy * sp,
            cy * cp
        ]
    }

    private func applySmile(_ expression: inout [Float], amount: Float) {
        func add(_ point: Int, _ axis: Int, _ value: Float) {
            expression[point * 3 + axis] += value
        }
        if amount > 0 {
            add(20, 1, -0.015 * amount)
            add(14, 1, -0.025 * amount)
            add(17, 1, 0.010 * amount)
            add(17, 2, 0.004 * amount)
            add(3, 1, -0.0045 * amount)
            add(7, 1, -0.0045 * amount)
        } else {
            add(14, 1, -0.020 * amount)
            add(17, 1, 0.003 * amount)
            add(19, 1, 0.020 * amount)
            add(19, 2, -0.005 * amount)
            add(20, 2, 0.010 * amount)
            add(3, 1, 0.0045 * amount)
            add(7, 1, 0.0045 * amount)
        }
    }
}
